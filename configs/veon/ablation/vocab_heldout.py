"""Use train_vocab.py --train-vocabulary to freeze the selected split inline."""
_base_ = ['./vocab_original.py']
model = dict(semantic_model=dict(vocabulary_file='work_dirs/vocab_split/train_vocabulary.json'))
