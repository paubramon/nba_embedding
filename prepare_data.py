import pandas as pd
import re


def remove_special_symbols(sentence):
    sentence = sentence.replace(u'\u2018', "'")  # left single quote mark
    sentence = sentence.replace(u'\u2019', "'")  # right single quote mark
    sentence = sentence.replace(u'\u201C', '"')  # left double quote mark
    sentence = sentence.replace(u'\u201D', '"')  # right double quote mark
    sentence = sentence.replace(u'\u2010', "-")  # hyphen
    sentence = sentence.replace(u'\u2011', "-")  # non-break hyphen
    sentence = sentence.replace(u'\u2012', "-")  # figure dash
    sentence = sentence.replace(u'\u2013', "-")  # dash
    sentence = sentence.replace(u'\u2014', "-")  # some sorta dash
    sentence = sentence.replace(u'\u2015', "-")  # long dash
    sentence = sentence.replace(u'\u2017', "_")  # double underscore
    sentence = sentence.replace(u'\u2014', "-")  # some sorta dash
    sentence = sentence.replace(u'\u2016', "|")  # long dash
    sentence = sentence.replace(u'\u2024', "...")  # ...
    sentence = sentence.replace(u'\u2025', "...")  # ...
    sentence = sentence.replace(u'\u2026', "...")  # ...
    sentence = sentence.replace(u'\u200f', "")  # ...
    sentence = sentence.replace(u'\u200b', "")  # ...
    sentence = sentence.replace(u'\u202a', "")  # ...

    # sentence = sentence.replace("\xce\x9d\xce\x91\xce\xa4\xce\x9f",u'NATO') # NATO

    sentence = sentence.replace(u'\u0391', "A")  # Greek Capital Alpha
    sentence = sentence.replace(u'\u0392', "B")  # Greek Capital Beta
    # sentence = sentence.replace(u'\u0393',"") # Greek Capital Gamma
    # sentence = sentence.replace(u'\u0394',"") # Greek Capital Delta
    sentence = sentence.replace(u'\u0395', "E")  # Greek Capital Epsilon
    sentence = sentence.replace(u'\u0396', "Z")  # Greek Capital Zeta
    sentence = sentence.replace(u'\u0397', "H")  # Greek Capital Eta
    # sentence = sentence.replace(u'\u0398',"") # Greek Capital Theta
    sentence = sentence.replace(u'\u0399', "I")  # Greek Capital Iota
    sentence = sentence.replace(u'\u039a', "K")  # Greek Capital Kappa
    # sentence = sentence.replace(u'\u039b',"") # Greek Capital Lambda
    sentence = sentence.replace(u'\u039c', "M")  # Greek Capital Mu
    sentence = sentence.replace(u'\u039d', "N")  # Greek Capital Nu
    # sentence = sentence.replace(u'\u039e',"") # Greek Capital Xi
    sentence = sentence.replace(u'\u039f', "O")  # Greek Capital Omicron
    sentence = sentence.replace(u'\u03a1', "P")  # Greek Capital Rho
    # sentence = sentence.replace(u'\u03a3',"") # Greek Capital Sigma
    sentence = sentence.replace(u'\u03a4', "T")  # Greek Capital Tau
    sentence = sentence.replace(u'\u03a5', "Y")  # Greek Capital Upsilon
    # ssentence = sentence.replace(u'\u03a6',"") # Greek Capital Phi
    sentence = sentence.replace(u'\u03a7', "T")  # Greek Capital Chi
    # sentence = sentence.replace(u'\u03a8',"") # Greek Capital Psi
    # sentence = sentence.replace(u'\u03a9',"") # Greek Capital Omega

    sentence = sentence.replace(u'\u03b1', "a")  # Greek small alpha
    sentence = sentence.replace(u'\u03b2', "b")  # Greek small beta
    # sentence = sentence.replace(u'\u03b3',"") # Greek small gamma
    # sentence = sentence.replace(u'\u03b4',"") # Greek small delta
    sentence = sentence.replace(u'\u03b5', "e")  # Greek small epsilon
    # sentence = sentence.replace(u'\u03b6',"") # Greek small zeta
    # sentence = sentence.replace(u'\u03b7',"") # Greek small eta
    # sentence = sentence.replace(u'\u03b8',"") # Greek small thetha
    sentence = sentence.replace(u'\u03b9', "i")  # Greek small iota
    sentence = sentence.replace(u'\u03ba', "k")  # Greek small kappa
    # sentence = sentence.replace(u'\u03bb',"") # Greek small lamda
    sentence = sentence.replace(u'\u03bc', "u")  # Greek small mu
    sentence = sentence.replace(u'\u03bd', "v")  # Greek small nu
    # sentence = sentence.replace(u'\u03be',"") # Greek small xi
    sentence = sentence.replace(u'\u03bf', "o")  # Greek small omicron
    # sentence = sentence.replace(u'\u03c0',"") # Greek small pi
    sentence = sentence.replace(u'\u03c1', "p")  # Greek small rho
    sentence = sentence.replace(u'\u03c2', "c")  # Greek small final sigma
    # sentence = sentence.replace(u'\u03c3',"") # Greek small sigma
    sentence = sentence.replace(u'\u03c4', "t")  # Greek small tau
    sentence = sentence.replace(u'\u03c5', "u")  # Greek small upsilon
    # sentence = sentence.replace(u'\u03c6',"") # Greek small phi
    sentence = sentence.replace(u'\u03c7', "x")  # Greek small chi
    sentence = sentence.replace(u'\u03c8', "x")  # Greek small psi
    sentence = sentence.replace(u'\u03c9', "w")  # Greek small omega

    sentence = sentence.replace(u'\u0103', "a")  # Latin a with breve
    sentence = sentence.replace(u'\u0107', "c")  # Latin c with acute
    sentence = sentence.replace(u'\u010d', "c")  # Latin c with caron
    sentence = sentence.replace(u'\u0161', "s")  # Lation s with caron
    sentence = sentence.replace(u'\u00e9', "e")  #

    return sentence.strip()

def preprocess_sentence(sent):
    # Remove special symbols
    sent = re.sub("([!|\"@=?^*])", ' ', sent)
    # Add spaces for other characters
    sent = re.sub("([!|'@#$%&/()=?^*\-,.:;])", r' \1 ', sent)
    sent = sent.replace('  ', ' ')

    return sent.strip()


filenameroot = 'Data/data'
file_end = '.xlsx'
num_files = 17
files = [filenameroot + str(i + 1) + file_end for i in range(num_files)]

player = []
team = []
position = []
text_short = []
text_long = []

for file in files:
    data = pd.ExcelFile(file)
    df1 = data.parse('SheetJS')
    for index, row in df1.iterrows():
        if row['Text'][0] != '$':
            match = re.match(r'^(.*)\n[\s\t\n]*-(.*)-(.*)\n[\s\t\n]*(.*)\n[\s\t\n]*(.*)\n',
                             remove_special_symbols(row['Text']))
            # Get player name
            player_name = match.group(1).strip().lower()
            player.append(player_name)

            # Get position
            position_name = match.group(2).strip().lower()
            position.append(position_name)

            # Get team
            team_name = match.group(3).strip().lower()
            team.append(team_name)

            # Get short text
            text_s = match.group(4).strip().lower()
            text_short.append(preprocess_sentence(text_s))

            # Get long text
            text_l = match.group(5).strip().lower()
            text_long.append(preprocess_sentence(text_l))

            print('player: %s ; team: %s' % (player_name, team_name))

final_data = pd.DataFrame(
    {'player': player, 'position': position, 'team': team, 'short_text': text_short, 'long_text': text_long})
final_data.to_csv('final_data.csv', encoding='latin-1')
