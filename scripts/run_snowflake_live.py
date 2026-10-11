"""Production Live Snowflake Execution and Validation Engine.

Executes Phase 4 against Snowflake:
1. Tests and establishes connection using externalbrowser SSO.
2. Executes DDL: Schema, Tables, Staging.
3. Loads staging data via internal stage and COPY INTO.
4. Executes stored procedures: SP_LOAD_CLV_SCORES and SP_REFRESH_BIAS_MONITOR.
5. Executes all 15 validation queries and records query IDs and results.
6. Executes idempotency re-run test.
7. Logs all actions and query results with zero fabrication.
"""

import sys
from pathlib import Path

# Add project root to sys.path
import os
import json
import time
from datetime import datetime, timezone
import pandas as pd
import snowflake.connector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PATHS

SNOWFLAKE_CONFIG = {
    "account": "BCNJRTY-LR91595",
    "user": "AKSHITA190904",
    "authenticator": "externalbrowser",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "CLV_DB",
    "schema": "CLV"
}


def run_live_pipeline():
    start_time = datetime.now(timezone.utc)
    log_records = []
    
    print("=" * 80)
    print("CLV TRUTH ENGINE - PHASE 4 LIVE SNOWFLAKE EXECUTION ENGINE")
    print(f"Timestamp (UTC): {start_time.isoformat()}")
    print("=" * 80)
    print("Connection Parameters:")
    for k, v in SNOWFLAKE_CONFIG.items():
        print(f"  {k}: {v}")
    print("-" * 80)
    
    conn = None
    connection_status = "FAILED"
    error_details = None
    
    try:
        print("Initiating live Snowflake connection via externalbrowser...")
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_CONFIG["account"],
            user=SNOWFLAKE_CONFIG["user"],
            authenticator=SNOWFLAKE_CONFIG["authenticator"],
            role=SNOWFLAKE_CONFIG["role"],
            warehouse=SNOWFLAKE_CONFIG["warehouse"]
        )
        connection_status = "CONNECTED"
        print(">>> LIVE CONNECTION SUCCESSFUL! <<<")
    except Exception as e:
        connection_status = "FAILED"
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        print(f"\n[LIVE CONNECTION FAILED]")
        print(f"Error Type: {error_details['error_type']}")
        print(f"Error Details: {error_details['error_message']}")
        print("-" * 80)
        
    execution_report = {
        "execution_timestamp_utc": start_time.isoformat(),
        "connection_parameters": SNOWFLAKE_CONFIG,
        "connection_status": connection_status,
        "error_details": error_details,
        "ddl_executed": False,
        "staging_loaded": False,
        "procedures_executed": False,
        "validation_queries": [],
        "idempotency_tested": False
    }
    
    if connection_status == "CONNECTED" and conn is not None:
        try:
            cur = conn.cursor()
            
            # 1. Fetch metadata
            cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_VERSION()")
            meta = cur.fetchone()
            execution_report["metadata"] = {
                "account": meta[0],
                "user": meta[1],
                "role": meta[2],
                "warehouse": meta[3],
                "version": meta[4]
            }
            print("Session Metadata:", execution_report["metadata"])
            
            # 2. Schema DDL
            print("Executing Schema DDL...")
            cur.execute("CREATE DATABASE IF NOT EXISTS CLV_DB")
            cur.execute("USE DATABASE CLV_DB")
            cur.execute("CREATE SCHEMA IF NOT EXISTS CLV")
            cur.execute("USE SCHEMA CLV")
            
            # Read and execute DDL scripts
            for script_name in ["02_create_clv_tables.sql", "03_create_staging.sql", "05_sp_load_clv_scores.sql", "06_sp_refresh_bias_monitor.sql"]:
                script_path = PATHS["project_root"] / "sql" / script_name
                print(f"Executing {script_name}...")
                with open(script_path, "r", encoding="utf-8") as f:
                    sql_text = f.read()
                # Split by semicolon where appropriate or execute procedure blocks
                cur.execute(sql_text)
            execution_report["ddl_executed"] = True
            
            # 3. Stage and load
            print("Staging and loading data...")
            scored_csv = PATHS["processed_data_dir"] / "snowflake_clv_scored_export.csv"
            bias_csv = PATHS["processed_data_dir"] / "snowflake_bias_monitor_export.csv"
            
            cur.execute(f"PUT file://{scored_csv.as_posix()} @CLV.STG_CLV_DATA_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
            cur.execute(f"PUT file://{bias_csv.as_posix()} @CLV.STG_CLV_DATA_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
            
            # Copy into staging
            load_script = PATHS["project_root"] / "sql" / "04_load_clv_scores.sql"
            with open(load_script, "r", encoding="utf-8") as f:
                load_sql = f.read()
            for stmt in load_sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            execution_report["staging_loaded"] = True
            
            # 4. Execute stored procedures
            print("Executing stored procedures...")
            cur.execute("CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4')")
            sp1_res = cur.fetchone()[0]
            print("SP_LOAD_CLV_SCORES result:", sp1_res)
            
            cur.execute("CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4')")
            sp2_res = cur.fetchone()[0]
            print("SP_REFRESH_BIAS_MONITOR result:", sp2_res)
            execution_report["procedures_executed"] = True
            execution_report["procedure_results"] = {
                "SP_LOAD_CLV_SCORES": sp1_res,
                "SP_REFRESH_BIAS_MONITOR": sp2_res
            }
            
            # 5. Validation queries
            print("Executing 15 validation queries...")
            val_script = PATHS["project_root"] / "sql" / "07_validation_queries.sql"
            with open(val_script, "r", encoding="utf-8") as f:
                val_sql = f.read()
            queries = [q.strip() for q in val_sql.split(";") if q.strip()]
            val_results = []
            for idx, q in enumerate(queries, 1):
                cur.execute(q)
                qid = cur.sfqid
                rows = cur.fetchall()
                val_results.append({
                    "test_number": idx,
                    "query_id": qid,
                    "sql": q,
                    "result": rows
                })
            execution_report["validation_queries"] = val_results
            
            # 6. Idempotency re-run test
            print("Executing idempotency re-run test...")
            cur.execute("CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4')")
            sp1_rerun = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM CLV.CLV_SCORED_OUTPUT WHERE run_id = 'run_20261011_phase4'")
            row_count_after_rerun = cur.fetchone()[0]
            execution_report["idempotency_tested"] = True
            execution_report["idempotency_result"] = {
                "sp_rerun_result": sp1_rerun,
                "row_count_after_rerun": row_count_after_rerun,
                "is_idempotent": (row_count_after_rerun == 2000)
            }
            
            cur.close()
            conn.close()
        except Exception as e:
            print("Error during execution:", e)
            execution_report["runtime_error"] = str(e)
            if conn:
                conn.close()

    # Save execution log and report
    out_json_path = PATHS["reports_dir"] / "snowflake_live_execution_manifest.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(execution_report, f, indent=2)
    print(f"\nSaved execution manifest: {out_json_path}")
    
    return execution_report


if __name__ == "__main__":
    run_live_pipeline()
