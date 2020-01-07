column_separator = "\t"

def load_csv_info(fname):
    star_rating_list = []
    review_body_list = []
    file=open(fname)
    file.readline()
    for line in file.readlines():
        marketplace, customer_id, review_id, product_id, product_parent,	product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date = line.strip().split(column_separator)
        star_rating_list.append(star_rating)
        review_body_list.append(review_body)
    return  star_rating_list, review_body_list

def load_text(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text


def split_reviews(star_rating_list, review):
    reviews_45 = []
    star_45 = []
    reviews_12 = []
    star_12 = []
    reviews1245 = []
    star1245 = []

    for ii, star in enumerate(star_rating_list):
        current_review = review[ii]
        if star == '4' or star == '5':
            reviews_45.append(current_review)
            star_45.append(star)
            reviews1245.append(current_review)
            star1245.append(star)
        elif star == '1' or star == '2':
            reviews_12.append(current_review)
            star_12.append(star)
            reviews1245.append(current_review)
            star1245.append(star)

    return reviews_12, reviews_45, reviews1245, star_12, star_45, star1245


