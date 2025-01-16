# // check the logest text in the file - true.csv and false.csv
# the text is in the `text` column

import pandas as pd

true = pd.read_csv('true.csv')
false = pd.read_csv('fake.csv')

true_text = true['text']
false_text = false['text']

# trim the text column
true_text = true_text.str.strip()
false_text = false_text.str.strip()
# check if the value in the text column is a empty string, and remove it
true_text = true_text[true_text != '']
false_text = false_text[false_text != '']


# calculate the number of words in the text column
true_text_len = true_text.str.split().apply(len)
false_text_len = false_text.str.split().apply(len)

print('True text max length:', true_text_len.max())
print('False text max length:', false_text_len.max())
print('True text min length:', true_text_len.min())
print('False text min length:', false_text_len.min())
print('True text mean length:', true_text_len.mean())
print('False text mean length:', false_text_len.mean())
print('True text median length:', true_text_len.median())
print('False text median length:', false_text_len.median())
print('True text std length:', true_text_len.std())
print('False text std length:', false_text_len.std())
print('True text var length:', true_text_len.var())
print('False text var length:', false_text_len.var())


# add to file the max text result:
with open('a.txt', 'w+') as f:
    f.write('True text max : {}\n'.format(true_text[true_text_len.idxmax()]))
    f.write('False text max : {}\n'.format(false_text[false_text_len.idxmax()]))
    f.write('True text min : {}\n'.format(true_text[true_text_len.idxmin()]))
    f.write('False text min : {}\n'.format(false_text[false_text_len.idxmin()]))