"""Tests connection to live Snowflake account."""
import sys
import snowflake.connector


def verify_connection():
    print("Initiating Snowflake connection test...")
    print("Account: BCNJRTY-LR91595")
    print("User: AKSHITA190904")
    print("Authenticator: externalbrowser")
    print("Role: ACCOUNTADMIN")
    print("Warehouse: COMPUTE_WH")

    try:
        conn = snowflake.connector.connect(
            account="BCNJRTY-LR91595",
            user="AKSHITA190904",
            authenticator="externalbrowser",
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH"
        )
        print("Connection established successfully!")
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_VERSION()")
        row = cur.fetchone()
        print("CURRENT_ACCOUNT:", row[0])
        print("CURRENT_USER:", row[1])
        print("CURRENT_ROLE:", row[2])
        print("CURRENT_WAREHOUSE:", row[3])
        print("CURRENT_VERSION:", row[4])
        cur.close()
        conn.close()
        print("Connection closed cleanly.")
        return True
    except Exception as e:
        print(f"Snowflake connection error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = verify_connection()
    if not success:
        sys.exit(1)
