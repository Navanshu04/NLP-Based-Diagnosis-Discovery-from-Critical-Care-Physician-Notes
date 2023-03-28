import pandas as pd

from scipy_wrapper import SWrapper

# nltk.download('punkt')

notes = pd.read_csv("C:/Users/Asus/Desktop/notes_physician_texttrim.csv")
print("Data dimensions:", notes.shape)
random_seed = 42
notes = notes.sample(n=400, random_state=random_seed)

def get_noun_chunks(statement):
    spacy_obj = SWrapper(statement)
    return spacy_obj.get_noun_chunks()


trim_text = notes["TEXT_TRIM"]
new_series = trim_text.apply(get_noun_chunks)
notes["TEXT_TRIM_NOUN_PHRASES"] = new_series
print(new_series.iloc[0])
notes.to_csv("C:/Users/Asus/Desktop/notes_physician_texttrim_np.csv")
# print(get_noun_chunks(trim_text.iloc[1]))
