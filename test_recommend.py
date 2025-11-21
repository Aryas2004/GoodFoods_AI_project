# test_recommend.py
import sqlite3, pprint

def sim_recommend(conn, party_size=2, cuisine=None, area=None, price=None):
    q = "SELECT id,name,cuisine,capacity,area,price_bucket,rating,amenities FROM restaurants WHERE capacity>=? "
    params = [party_size]
    if cuisine:
        q += "AND cuisine LIKE ? "
        params.append(f"%{cuisine}%")
    if area:
        q += "AND area LIKE ? "
        params.append(f"%{area}%")
    if price:
        q += "AND price_bucket=? "
        params.append(price)
    q += "ORDER BY rating DESC LIMIT 10"
    cur = conn.execute(q, params)
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    return rows

if __name__ == "__main__":
    conn = sqlite3.connect("goodfoods.db")
    print("Count restaurants:", conn.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0])
    print("\nSample rows (first 10):")
    for r in conn.execute("SELECT id,name,cuisine,capacity,area,price_bucket,rating FROM restaurants LIMIT 10"):
        print(r)
    print("\nSimulate strict recommend(2,'italian','bandra','mid') ->")
    pp = sim_recommend(conn, 2, "italian", "bandra", "mid")
    pprint.pprint(pp)
    print("\nSimulate fallback recommend(2,None,None,None) -> top-rated fallback")
    pp2 = sim_recommend(conn, 2, None, None, None)
    pprint.pprint(pp2[:5])
    conn.close()
