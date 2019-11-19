# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:43:45 2019

@author: Mar Adrian
"""
import nltk ,re 
from nltk import word_tokenize
from nltk.corpus import stopwords
r= "I found \Black Passenger / Yellow Cabs\ to be a very readable book.  It was fun going hearing about Mr. Bryas sexual escapades- at first.  It was refreshing to have someone who could frequently be very perceptive speak honestly not only about his life, but the sexual escapades and shenanigans he was engaging in.  And to link that behavior with the horrific conditions of his early youth in a ghetto in Jamaica, and more loosely, with the Jamaican cultural socialization which influenced him.  These elements humanized and explained some of his less than noble behavior.<br /><br />However, what becomes extremely obvious as the book goes on (and on and on it does go)is that, even in Mr. Bryas own words, he suffers fromdissocial personality disorder or some such thing.  Perilously close, it seems to me, to sociopathic behavior.  Even the title, in retrospect, speaks to the sociopaths view of humanity.  Mr. Bryan is apassenge, i.e. a person, the woman arecab- nohumans.<br /><br />The irony, of course, for anyone whs read the novel is that Mr. Bryan frequently speaks of how the Japanese culture traumatizes and demeans women.  However, he is a flagrant abuser of women- though he frequently presents himself as the opposite, as one who has nutured these poor women back to health!<br /><br />He has impregnated more than 10 women while in Japan- and in most case paid or helped to pay for the abortions.  More than 10 women!  All because he enjoys the feel of sex without a condom.  And because he knows that most of the women wot insist that he wear one due to their timidity!  So, in effect, hd have woman after woman go through the greater trauma of an abortion; and have fetus be terminated (some would even say murdered); and he would even have to pay (in some cases only half) for his temporary pleasure!  This, more than ten times!<br /><br />And let us not forget how he then has his cake and eats it too as he makes out how horrible it is of the Japanese to run an abortion industry, instead of having oral contraceptives more readily available.<br /><br />He also even refers to the women in his writings as cars.  He prefers, he tells the reader, his \bentley\ (Shoko) but theferrar (Azusa) is sexier and her parents are so accepting of him!<br /><br />The language he uses time and again, in conjunction with the repeated behaviors, indicates that closeness he shares with the sociopath: people are things to be used, manipulated.  People are not people: they are not truly human with feelings and the right to be treated asends in themselve- but rather they are trappings, are thigns, to be used and manipulated by the sociopath.<br /><br />Mr. Bryan frequently talks about how he doest really love- indeed, it seems that he doest even like very much- Azusa.  But shs so good looking that he wants to have children with her; while, of course, marrying Shoko (or perhaps converting to Islam- though hs an atheist- in the hopes of being able to legally marry both of them).<br /><br />This goes on and on.<br /><br />Again, at first, much of this seemed fun, interesting, a man with hardbreaks coming into his own.  However, this behavior is- and has been- going on for years, even into his forites.<br /><br />Yes, thers theaddiction however, I got the distinct impression that Mr. Bryan was being more honest than perhaps he meant to be when he on repeated occasions told the reader about his personality disorder.  He truly has one, it seems.  He may not be a sociopath, but in many respects hisdisorde is such that other people are things.  For me, to get behind the eyes, so to speak, of a person like that is a frightening experience.  But, pun intended, aneyopene!<br /><br />Is also curious that Mr. Bryan has, it seems, only female friends.  In part, he would have the reader believe it is because of his greater sensitivity to females and their plight.  He sometimes refers to himself as alesbian in a mals body  His sensitivity is hogwash!  Woman form the perfect dovetail for the likes of him:  one, he likestai and two, he knows that he can manipulate them far more easily than he can men.  These two features make them delectable to him.<br /><br />A minor note.  Much of the book is fairly well written with this curious feature:  there are numerous ratherloft words sprinkled throughout the text.  For example he writes of one woman \she tried corybanticly to cloak herself with anything in sight\ (Yukari; p331).  Now, I have a mastes degree in English, but I dot know what the freak \corybanticly\ means.  The point is that too frequently this text is larded with extremely high falutin words, amid, of course, the booty calls, and his petrified rod, etc.<br /><br />Most curious!<br /><br />Anyway, this is in many ways an interesting read: some of the incidents are enjoyable to read; much of the writing is actually pretty good; there are some sharp insights into Japanese culture ( I lived in Japan for a year so I had some access to what hs referring to); thers great awareness generally of culture and its influence (Jamaican, American, Japanese).  Thers the fascinating other perspective- both replusive and fascinating- of the view from thepersonality disorde.<br /><br />Anyway, I give this book 3 stars.<br /><br />I hope this review helped.<br /><br />Aloha,<br />paul"

###PUNCTUATION###
import string
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

rp=remove_punctuation(r)

###NUMBERS INTO WORDS###
# import the inflect library 
import inflect 
p = inflect.engine() 

# convert number into words 
def convert_number(text): 
	# split string into list of words 
	temp_str = text.split() 
	# initialise empty list 
	new_string = [] 

	for word in temp_str: 
		# if word is a digit, convert the digit 
		# to numbers and append into the new_string list 
		if word.isdigit(): 
			temp = p.number_to_words(word) 
			new_string.append(temp) 

		# append the word as it is 
		else: 
			new_string.append(word) 

	# join the words of new_string to form a string 
	temp_str = ' '.join(new_string) 
	return temp_str 

rpp=convert_number(rp)

###WITHOUT STOPWORDS###
r_w=word_tokenize(rpp)
def sw(text):
    stop_en = stopwords.words('english')
    no = [w for w in text if w not in stop_en]
    return no 
r_sw=sw(r_w)

###WITHOUT LOWER CASE###
def text_lowercase(text): 
    l = [item.lower() for item in text]
    return l 
r_lc=text_lowercase(r_sw)







    
    
    
    