# Data

`financial_phrasebank_75agree.csv` contains the 3,453 sentences of the Financial PhraseBank on which at least 75% of annotators agreed, with columns `text` and `label` (`negative`, `neutral` or `positive`). Rebuild it with `python -m scripts.prepare_data`, which downloads the original archive and checks its SHA-256.

**Source.** P. Malo, A. Sinha, P. Korhonen, J. Wallenius and P. Takala (2014), "Good debt or bad debt: Detecting semantic orientations in economic texts", *Journal of the Association for Information Science and Technology* 65(4). Distributed via [huggingface.co/datasets/takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).

**Licence.** [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The CSV here is a format conversion of the original file (Latin-1 `sentence@label` lines to UTF-8 CSV) and is shared under the same licence. Non-commercial use only.

**What the labels mean.** Annotators with finance backgrounds labelled each sentence by its likely effect on the company's share price from an investor's point of view. "Neutral" therefore includes many factual sentences, which is why it is the largest class (62%).
