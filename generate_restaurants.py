# generate_restaurants.py
import csv
import random
from datetime import datetime
cuisines = ["Indian","Italian","Chinese","Continental","Mexican","Thai","Japanese","Mediterranean","Cafe","Seafood"]
areas = ["Andheri","Bandra","Lower Parel","Colaba","Koramangala","Indiranagar","MG Road","Noida Sector 18","Connaught Place","Jayanagar"]
price_buckets = ["low","mid","high"]
amenities_pool = ["wifi","parking","outdoor_seating","wheelchair_access","live_music","private_dining"]
def gen_restaurant(i):
    name = f"GoodFoods_{i}"
    cuisine = random.choice(cuisines)
    capacity = random.randint(20, 200)
    area = random.choice(areas)
    price = random.choice(price_buckets)
    rating = round(random.uniform(3.0,4.9),1)
    amenities = ",".join(random.sample(amenities_pool, k=random.randint(1,3)))
    return [i, name, cuisine, capacity, area, price, rating, amenities]
rows = [gen_restaurant(i) for i in range(1,101)]
with open("restaurants.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id","name","cuisine","capacity","area","price_bucket","rating","amenities"])
    w.writerows(rows)
print("Generated restaurants.csv with", len(rows))
