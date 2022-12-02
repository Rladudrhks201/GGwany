# MultiLabelBinarizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer

multiclass_feature = [("texas","florida"),
                        ("california","alabama"),
                        ("texas","florida"),
                        ("delware","florida"),
                        ("texas","florida")]
# print(multiclass_feature)                    

one_hot_multiclass = MultiLabelBinarizer() # 다중 클래스 원 핫 인코더 객체 생성
one_hot_multiclass.fit_transform(multiclass_feature) # 다중 클래스 특성을 원 핫 인코더 실행

one_hot_multiclass_classes = one_hot_multiclass.classes_
print(one_hot_multiclass_classes)

print(one_hot_multiclass.fit_transform(multiclass_feature))