from filesplit.merge import Merge

merge = Merge('./separated_model', '.', 'model.h5')
merge.merge()