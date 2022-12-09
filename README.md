# RuTransform

RuTransform is a Python framework for adversarial attacks and text data augmentation for Russian.

#### Table of contents

- [Setup & Usage](https://github.com/RussianNLP/rutransform/#setup--usage)
- [Examples](https://github.com/RussianNLP/rutransform/#examples)
  - Dataset Transformation on sample data
  - Dataset Transformation on your own data
  - Custom Constraints
  - Sentence Transformations
  - Custom Transformations
- [Framework Design](https://github.com/RussianNLP/rutransform/#framework-design)
- [Cite us](https://github.com/RussianNLP/rutransform/#cite-us)

## Setup & Usage

#### Requirements

Python >= 3.7

### Installation

```
git clone https://github.com/RussianNLP/rutransform
cd rutransform
pip install .
```

### Examples

<details>
    <summary><b>Dataset Transformation (sample data)</b></summary>

For the full list of supported transformations with examples see [supported transformations](https://github.com/RussianNLP/rutransform/#supported-transformations).

```
import pandas as pd
from rutransform.transformations import DatasetTransformer
from rutransform.utils.args import TransformArguments


# load data
dataset = pd.read_json('test_data/worldtree.json', lines=True)

# load arguments
transformation = "butter_fingers"
probability = 0.3
args = TransformArguments(transformation=transformation, probability=probability)

# init dataset transformer
tr = DatasetTransformer(
    dataset=dataset,
    text_col='question',
    task_type='multichoice_qa',
    args=args,
    return_type='pd'  # format of the resulting dataset (default is 'hf')
)

# run transformation
output = tr.transform()
```
`DatasetTransformer` outputs a named tuple with dataset similarity scores:
```
score = output.score  # mean dataset similarity score: 0.9146944761276246
scores = output.scores  # similarity scores for each sentence: array([0.93971652, 0.94295949, 0.8272841 , 0.98828816, 0.87522411])
std = output.std  # std of the similarity scores: 0.05663837594035781
```
...and the transformed dataset:
```
transformed_data = output.transformed_dataset

print('Original data:\n',  dataset['question'][0])
print('Transformed data:\n', transformed_data['question'][0])
```
```
Original data:
 Когда мороженое не кладут в морозильную камеру, мороженое превращается из ___. (A) твердого тела в газ (B) газа в жидкость (C) твердого тела в жидкость (D) жидкости в газ
Transformed data:
 Когда мороженое не кладут в морозильную камеру, мороженое превращается из ___. (A) твердого тела в газ (B) газа в жидвость (C) мвердого тела в дидкость (D) жидкости в гкз
```

</details>

<details>
    <summary><b>Dataset Transformation (own data)</b></summary>


RuTransform can easily be adapted to other tasks. To use the framework on your own data, simply specify the text (`text_col`) and/or target (`label_col`) column names and choose the suitable constraints (pass them into the `custom_constraints` argument). For example, to run transformation on the [DaNetQA](https://russiansuperglue.com/tasks/task_info/DaNetQA) data [(Shavrina et al,. 2020)](https://aclanthology.org/2020.emnlp-main.381/) we choose to perturb the `passage` text and use the `NamedEntities` constraint to preserve proper nouns:

```
import pandas as pd
from rutransform.transformations import DatasetTransformer
from rutransform.utils.args import TransformArguments
from rutransform.constraints import NamedEntities

# load data
dataset = pd.read_json('test_data/danet_qa.json', lines=True)

# init arguments
transformation = "back_translation"
probability = 0.5
args = TransformArguments(transformation=transformation, probability=probability)

# init dataset transformer
tr = DatasetTransformer(
    dataset=dataset,
    text_col='passage',
    args=args,
    return_type='pd',
    custom_constraints=[NamedEntities()],
    device='cuda:0'
)

# run transformation
output = tr.transform()

print('Original data:\n', dataset['passage'][4],)
print('Transformed data:\n', output.transformed_dataset['passage'][4])
```
```
Original data:
 Средство коммуникации. В своей простейшей форме искусство представляет собой средство коммуникации. Как и большинство прочих способов коммуникации, оно несет в себе намерение передать информацию аудитории. Например, научная иллюстрация — тоже форма искусства, существующая для передачи информации. Ещё один пример такого рода — географические карты. Однако содержание послания не обязательно бывает научным. Искусство позволяет передавать не только объективную информацию, но и эмоции, настроение, чувства.
Transformed data:
 Средство коммуникации. В своей простой форме искусство является средством общения. Как и большинство других средств коммуникации, она намерена доводить информацию до сведения аудитории. Например, научная иллюстрация — тоже форма искусства, существующая для передачи информации. Ещё один пример такого рода — географические карты. Однако содержание послания не обязательно бывает научным. Искусство позволяет передавать не только объективную информацию, но и эмоции, настроение, чувства.
```
</details>

<details>
    <summary><b>Custom Constraints</b></summary>

If the provided constraints are not enough, you can create your own ones by simple  class inheritance. For example, to run transformation on the [RWSD](https://russiansuperglue.com/tasks/task_info/RWSD) dataset [(Shavrina et al,. 2020)](https://aclanthology.org/2020.emnlp-main.381/), we create an `RWSDConstraint`:

```
from rutransform.constraints import Constraint
from rutransform.constraints.utils import parse_reference
from typing import List, Optional
from spacy.language import Language

class RWSDConstraint(Constraint):
    def __init__(self, target_col_name: str, reference_key: str, noun_key: str) -> None:
        super().__init__(name='rwsd_constraint')
        self.target_col_name = target_col_name
        self.reference_key = reference_key
        self.noun_key = noun_key
    
    def patterns(
        self, text: Optional[dict], spacy_model: Optional[Language]
    ) -> List[List[dict]]:
        morph = parse_reference(text[self.target_col_name][self.noun_key], spacy_model)
        antecedent_feats = list(morph.values())
        patterns = [
            [{"TEXT": {"IN": text[self.target_col_name][self.reference_key].split() + text[self.target_col_name][self.noun_key].split()}}],
            [{"POS": {"IN": ["NOUN", 'PROPN']}, "MORPH": {"IS_SUPERSET": antecedent_feats}}],
        ]
        return patterns
```
To use custom constraints during the transformation, pass them into the `custom_constraints` argument:
```
import pandas as pd
from rutransform.transformations import DatasetTransformer
from rutransform.utils.args import TransformArguments

# load data
dataset = pd.read_json('test_data/rwsd.json', lines=True)

# load arguments
transformation = "eda"
probability = 0.5
args = TransformArguments(transformation=transformation, probability=probability)

# init dataset transformer
tr = DatasetTransformer(
    dataset=dataset,
    text_col='text',
    args=args,
    custom_constraints=[
        RWSDConstraint(
            target_col_name='target', reference_key='span2_text', noun_key='span1_text'
        )
    ],
    return_type='pd'  # format of the resulting dataset (default is 'hf')
)

# run transformation
output = tr.transform()

print('Target:', dataset['target'][0]) 
print('Original data:\n', dataset['text'][0],)
print('Transformed data:\n', output.transformed_dataset['text'][0])
```
```
Target: {'span1_text': 'статью', 'span2_text': 'читает ее', 'span1_index': 7, 'span2_index': 9}
Original data:
 Сара взяла в библиотеке книгу, чтобы написать статью. Она читает ее, когда приходит с работы.
Transformed data:
 Сара книгу , чтобы написать статью Она читает ее с работы .
```

</details>

<details>
    <summary><b>Sentence Transformation</b></summary>

All of the transformations, supported by the framework, can be applied not only to the while datasets, but sentences alone.

```
from rutransform.transformations import (
    SentenceAdditions,
    ButterFingersTransformation, 
    EmojifyTransformation,
    ChangeCharCase,
    BackTranslationNER,
    Paraphraser,
    RandomEDA,
    BAE
)

# initialize the transformations arguments, but you can leave out the transformation
args = TransformArguments(probability=0.5)

# transform the sentence
tr = SentenceAdditions(args=args)
tr.generate('мама мыла раму')
```

```
['мама мыла раму, Мама мыла раму,']
```

```
tr = ButterFingersTransformation(args=args,)
tr.generate('мама мыла раму')
```
```
['ммаа мырв ламу']
```
</details>


<details>
    <summary><b>Custom Transformation</b></summary>

RuTransform allows one to create their own custom transformations. Here is the example of a simple transformation that randomises word order.


First, you need to define the `SentenceOperation` class for the transformation, which has `__init__` and `generate` functions. 

Note, that the function arguments must stay unchanged for further compatability with the framework. We also define a separate function for th transformation itself, to keep the code more readable.
    
```
import random
import spacy
from rutransform.transformations.utils import SentenceOperation
from typing import Optional, List, Union, Dict


def random_word_order(sentence, spacy_model, seed, max_outputs):
    
    """
    Randomise word order
    """
    
    random.seed(seed)
    
    if not spacy_model:
        spacy_model = spacy.load('ru_core_news_sm')
    
    tokens = [token.text for token in spacy_model(sentence)]
    
    return [' '.join(random.sample(tokens, k=len(tokens))) for _ in range(max_outputs)]
    

class RandomWordOrder(SentenceOperation):
    def __init__(
        self, args, seed=42,
        max_outputs=1, device="cpu",
        spacy_model=None,
):
        super().__init__(
            args=args,
            seed=seed,
            max_outputs=max_outputs,
            device=device,
            spacy_model=spacy_model,
        )
        
    def generate(
        self,
        sentence: str,
        stop_words: Optional[List[Union[int, str]]] = None,
        prob: Optional[float] = None,
    ) -> List[str]:
        
        transformed = random_word_order(
            sentence=sentence,
            seed=self.seed,
            spacy_model=self.spacy_model,
            max_outputs=self.max_outputs
        )

        return transformed
    
```

Now the transformation is ready to use on the sentence level:
    
```
from rutransform.utils.args import TransformArguments

args = TransformArguments()
tr = RandomWordOrder(args=args, max_outputs=5)
tr.generate("мама мыла раму")
    
```

```
['раму мама мыла',
 'раму мыла мама',
 'мама раму мыла',
 'раму мама мыла',
 'мама раму мыла']
```
    
After creating the transformation, you can add it to an existing Transformer, by simply inheriting the class and changing the `transform_info` fuction: 
    
```
from rutransform.transformations import EDATransformer


class EDATransformer(EDATransformer):
    def __init__(
        self,
        transformations: List[str],
        task_type: str,
        args: TransformArguments,
        text_col: Optional[str] = "text",
        label_col: Optional[str] = "label",
        seed: int = 42,
        device: str = "cpu",
        constraints=None,
    ) -> None:
        
        super().__init__(
            transformations=transformations,
            task_type=task_type,
            args=args,
            text_col=text_col,
            label_col=label_col,
            seed=seed,
            device=device,
            constraints=constraints
        )
    
    def transform_info() -> Dict[str, Optional[SentenceOperation]]:

        info = {"eda": RandomEDA, "word_order": RandomWordOrder}

        return info
```
    
...or create a Transformer from scratch by inheriting the `Transformer` class and defining several functions:

- `transform_info`: a staticmethod, must return a dictionary {transformation name: corresponding SentenceOperation class}. It is used to load the list of all the available transformations
- `_apply_transformation`: a function that applies the transformations to text until the transformed text passes the similarity threshold and returns a list of transformed texts and their similarity scores
- `transform` (optional): a function that takes a sentence as input and transforms it

For more information on the `Transformer` class and its structure see [here](rutransform/transformations/transformer.py).


Once you have created the Transformer, add it to the [rutransform/transformations/transformers](rutransform/transformations/transformers) folder and edit the [`__init__.py`](rutransform/transformations/__init__.py) file.

Now you transformation is ready for use!
</details>

##  Framework Design

### Supported Transformations

Following the generally accepted typology ([Zhang et al., 2020](https://arxiv.org/pdf/1901.06796.pdf); [Wang et al., 2021b](https://aclanthology.org/2022.naacl-main.339/)), we divide the transformations included in the framework in two types, depending on their target.

#### Word-Level Transformations

Word-level perturbations utilize several strategies to perturb tokens, ranging from imitation of typos to synonym replacement:

Type    | Transformation          | Paper/Source | Main Idea |Example                                                         |
:-------|:------------------------|:-------------|:----------|:------------------------------------------------------------------|
Spelling|ButterFingers (`butter_fingers`)          | [(Dhole, 2021)](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/butter_fingers_perturbation) | Adds noise to data by mimicking spelling mistakes made by humans through character swaps based on their keyboard distance | This is a se**m**tence **r**o test t**j**e code |
Spelling|Case (`case`)                   | [(Z. Wang, 2021)](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/change_char_case) | Adds noise to data through case alteration | This is a sentence to tes**T** t**H**e c**OD**e |
Modality|Emojify (`emojify`)                | [(Wang, 2021)](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/emojify)| Replaces the input words with the corresponding emojis, preserving their original meaning | This is a sentence to test the 👩‍💻 |
Context |BAE (`bae`)| [(Garg and Ramakrishnan, 2020)](https://arxiv.org/abs/2004.01970) | Inserts and replaces tokens in the original text by masking a portion of the text and using masked LMs to generate contextually appropriate words | This is a sentence to test the **given** code |

#### Sentence-Level Transformations
In contrast to word-level perturbations, sentence-level perturbation techniques affect the syntactic structure:

Type       |Transformation          | Paper/Source | Main Idea |Example                                                            |
:----------|:-----------------------|:-------------|:----------|:------------------------------------------------------------------|
Random     |EDA<sub>DELETE</sub> (`eda`)| [(Wei and Zou, 2019)](https://aclanthology.org/D19-1670.pdf) | Randomly deletes tokens in the text | This **_** a sentence to test the code |
Random     |EDA<sub>SWAP</sub> (`eda`)  | [(Wei and Zou, 2019)](https://aclanthology.org/D19-1670.pdf) | Randomly swaps tokens in the text | **code** is a sentence to test the **This** |
Paraphrasis|Paraphraser (`paraphraser`)           | [(Fenogenova, 2021)](http://bsnlp.cs.helsinki.fi/papers-2021/2021.bsnlp-1.2.pdf) | Generates variations of the context via simple paraphrasing | **I am just testing the code** |
Paraphrasis|BackTranslation (`back_translation`)       | [(Yaseen and Langer, 2021)](https://arxiv.org/abs/2108.11703) | Generates variations of the context through back-translation (ru -> en  -> ru) |**This sentence tests the code** |
Distraction|AddSent (`add_sent`)               | [(Jia and Liang, 2017)](https://aclanthology.org/D17-1215/) | Generates extra words or a sentence at the end of the text. For multiple choice QA tasks it replaces one or more choice options with a generated one | This is a sentence to test the code **, if you want to delete it** |

The examples are given in English for illustration purposes.

### Probability thresholds

The degree of the input modification can be controlled with an adversarial probability threshold, which serves as the hyperparameter. The higher the probability, the more the input gets modified. 

### Constraints

The RuTransform's attacks and perturbations do not drastically change the input's meaning. Despite this, we consider the use of rule-based constraints that keep the linguistic structure and task-specific aspects unchanged. For instance, it is crucial to leave named entities in the QA tasks untouched and not modify the syntactic structure and anaphors when perturbing the coreference resolution task examples.

Name| Description | Additional Requirements | Example |
:---|:------------|:------------------------|:--------|
`Jeopardy` | Jeopardy type conatraints, including (1) Noun Phrases such as THIS FILM, THIS ACTOR, both UPPER and lower cased, (2) 'X', (3) «Named Entity in parentheses» | - | For the first time, **THIS soda** appeared in 1958 in Spain, the name of the drink is translated from the Esperanto language as **“amazing”**.|
`NamedEntities`|Matches all the named entities in text| - |The singer from **Turkey** who impressed us all.|
`Multihop`| Constraints for multihop QA tasks. Matches all the bridge and main answers important for hops | - | `Question:` Where is the source of the river, the tributary of which is the Getar, located? `Supporting Text:` The **Getar** is a river in Armenia. It originates in the Kotayk region, flows through the central part of Yerevan and flows into **the Hrazdan**. `Main Text:` **The Hrazdan**, a river in Armenia, is the left tributary of the Aras. It originates at the northwest extremity of Lake **Sevan**, near the city of **Sevan**. `Answer:` Sevan |
`Referents` | Constraints for coreference resolution tasks. Matches (1) the anaphoric pronoun, (2) all possible antecedents (3) all verbs referring to antecedents and anaphor | Markup of the possible antecedents and anaphors | The **singer** from **Turkey** **who** **impressed**  us all.|

### Semantic filtering

We follow [Wang et al., 2021](https://arxiv.org/abs/2111.02840) on filtering the adversarial examples with BERTScore [(Zhang et al., 2019)](https://arxiv.org/abs/1904.09675), a BERT-based text similarity metric [(Devlin et al., 2019)](https://aclanthology.org/N19-1423.pdf). We measure the semantic similarity between the original input and adversarial output and keep examples with the highest similarity score. In cases when the score is lower than a specified threshold, we iteratively decrease the adversarial probability threshold and re-score the new adversarial examples. 


## Cite us

```
@article{taktasheva2022tape,
  title={TAPE: Assessing Few-shot Russian Language Understanding},
  author={Taktasheva, Ekaterina and Shavrina, Tatiana and Fenogenova, Alena and Shevelev, Denis and Katricheva, Nadezhda and Tikhonova, Maria and Akhmetgareeva, Albina and Zinkevich, Oleg and Bashmakova, Anastasiia and Iordanskaia, Svetlana and others},
  journal={arXiv preprint arXiv:2210.12813},
  year={2022}
}
```

## License

All the code is available under the Apache 2.0 license.
