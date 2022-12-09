# Transformations

## Word-Level Transformations

Type    | Transformation          | Example                                                         |
:-------|:------------------------|:----------------------------------------------------------------|
Spelling|`ButterFingers`          | This is a se**m**tence **r**o test t**j**e code                 |
Spelling|`Case`                   | This is a sentence to tes**T** t**H**e c**OD**e                 |
Modality|`Emojify`                | This is a sentence to test the 👩‍💻                               |
Context |`BAE`<sub>`INSERT`</sub> | This is a sentence to test the **given** code                   |
Context |`BAE`<sub>`REPLACE`</sub>| This is a sentence to check the code                            |

## Sentence-Level Transformations

Type       |Transformation          | Example                                                            |
:----------|:-----------------------|:-------------------------------------------------------------------|
Random     |`EDA`<sub>`DELETE`</sub>| This **_** a sentence to test the code                             |
Random     |`EDA`<sub>`SWAP`</sub>  | **code** is a sentence to test the **This**                        |
Paraphrasis|`Paraphraser`           | **I am just testing the code**                                     |
Paraphrasis|`BackTranslation`       | **This sentence tests the code**                                   |
Distraction|`AddSent`               | This is a sentence to test the code **, if you want to delete it** |
