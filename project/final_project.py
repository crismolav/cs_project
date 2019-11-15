column_separator = "\t"

star_rating_list = []
review_body_list= []

def load_csv_info(fname="prueba2.txt"):
    file=open(fname)
    file.readline()
    for line in file.readlines():
        marketplace, customer_id, review_id, product_id, product_parent,	product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date = line.strip().split(column_separator)
        star_rating_list.append(star_rating)
        review_body_list.append(review_body)
    return  star_rating_list, review_body_list

star_rating_list, review_body_list = load_csv_info(fname="prueba2.txt")
print(star_rating_list)