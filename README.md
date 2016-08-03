# LSA (Latent Semantic Analysis) 

This is a python implementation of Latent Semantic Analysis.

Support both English and Chinese.

# Usage

You can run it in cmd.

```
python lsa.py [-h] [-m MODELDIRECTORY] [-o OUTPUTDIRECTORY] [-s STOPWORDSFILEPATH] [-d DIMENSION] mode documentsFilePath
```

There are two modes : train and infer. 

The former is used to train the model with train data and the latter is used to infer the most similar documents in the training data of test data.  

# Example

## train

```
python lsa.py train example/dataset.txt -m example/model -o example/output -s example/stopwords.txt
```

You will get the coordinates of documents and words and a visualization(using only the first 2 dimensions) in the output directory you specify.

![visualization](https://github.com/laserwave/LSA/blob/master/example/output/visualization.png)

## infer

```
python lsa.py infer example/testdata.txt -m example/model -o example/output
```

The results will be in the file infer.txt in the output directory you specify.

The results of the example is shown as:

11(0.0346613089274) ,5(0.0405232014515) ,8(0.0532691450851) ,10(0.156756453557) ,12(0.18266734132)

4(0.00510175786639) ,6(0.00645403595639) ,7(0.0120286431551) ,8(0.235187842506) ,11(0.253795678664)

The two lines are the results of the two test documents.

The five numbers (each line) are the numbers(counting from zero) of the 5 most similar documents in the training set.

The number in the bracket is the angle of two vectors(unit is radian), representing the test document and the corresponding document in the training set.

Author
============

 * ZhikaiZhang 
 * Email <zhangzhikai@seu.edu.cn>
 * Blog <http://zhikaizhang.cn>
 * [自然语言处理之LSA](http://zhikaizhang.cn/2016/05/31/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B9%8BLSA/)