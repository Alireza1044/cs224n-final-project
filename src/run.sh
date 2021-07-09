cd 1.word2vec
python run.py
python run.py --char dwight
python query.py
python query.py --char dwight

cd ../2.tokenization
python main.py

cd ../3.parsing
python run.py

cd ../4.language_model
python train.py
python train.py --char dwight
python predict.py --length 10 --input "they want us to"
python predict.py --length 10 --input "they want us to" --char dwight

cd ../5.fine_tuning
python main.py
python main.py --char dwight
python main.py --predict --length 20 --text "i am"
python main.py --predict --length 20 --text "i am" --char dwight
