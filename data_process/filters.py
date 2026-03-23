from collections import Counter
import string

# True to drop
def ngram_repeat_ratio(tokens, n=5, threshold=0.30):
    if len(tokens) < n:
        return False
    ngrams = (tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    counter = Counter(ngrams)
    total = sum(counter.values())
    unique = len(counter)
    repeat_ratio = 1 - unique / total
    return repeat_ratio > threshold


def filter_punctuation_ratio(text, threshold: float = 0.30) -> bool:
    punct_count = sum(1 for c in text if c in string.punctuation)
    return (punct_count / len(text)) > threshold


if __name__ == "__main__":
    text = """The United Kingdom's membership in the European Union (EU) from 1973 until 2020 brought a complex set of benefits and challenges. Here is a structured overview of the **pros** and **cons** of the UK being part of the EU:

---

## **Pros of the UK Being Part of the EU**

### 1. **Economic Integration**
- **Access to the Single Market:** UK businesses could trade freely with the other 27 EU member states without tariffs, quotas, or border checks, boosting exports and imports.
- **Investment and Growth:** EU membership attracted foreign direct investment (FDI), especially in manufacturing, finance, and services.
- **EU Funding:** The UK received financial support from the EU for infrastructure, agriculture, and regional development, although it contributed more in taxes than it received.

### 2. **Freedom of Movement**
- **Labor Mobility:** British citizens could work, live, and study in other EU countries, and vice versa, which helped industries like healthcare, agriculture, and hospitality.
- **Students and Researchers:** Easier access to education and research opportunities across Europe for UK students and scientists.

### 3. **Political and Diplomatic Influence**
- **Global Voice:** As part of the EU, the UK had a stronger political voice in global affairs through the EU's foreign policy and collective bargaining power.
- **EU-Led Diplomacy:** Participation in EU-led initiatives on climate change, trade agreements, and security.

### 4. **Regulatory and Standards Harmonization**
- **EU Regulations:** A level playing field for all members, reducing the need for UK businesses to comply with separate national regulations.
- **Consumer Protection:** High standards in areas like food safety, product quality, and environmental regulation.

### 5. **Security and Cooperation**
- **EU Security Cooperation:** Participation in the Schengen Area (until Brexit), shared intelligence, and joint counter-terrorism efforts.
- **Justice and Home Affairs:** Cooperation in areas like policing, border control, and judicial matters.

### 6. **Cultural and Social Benefits**
- **Cultural Exchange:** Closer ties with other European countries for art, music, literature, and festivals.
- **EU Citizenship:** UK citizens had the right to vote in European Parliament elections and run for office.

---

## **Cons of the UK Being Part of the EU**

### 1. **Loss of Sovereignty**
- **Limited Control:** The UK had less control over laws, regulations, and policies, especially in areas like agriculture, fisheries, and environmental regulation.
- **EU Court of Justice:** UK courts had to follow rulings from the Court of Justice of the EU, which some saw as undemocratic.

### 2. **Economic Constraints**
- **Red Tape:** Compliance with EU regulations and bureaucratic procedures added costs for businesses.
- **Tightened Regulations:** Many EU rules were seen as burdensome, especially for"""
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    # tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    # print(tokens)
    # print(len(tokens))
    print(ngram_repeat_ratio(text, n=5, threshold=0.28))
    # print(filter_punctuation_ratio(text, threshold=0.45))
