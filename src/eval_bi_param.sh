python -u eval.py --gt ../corpus/news/sina_test.pkl --inputs ../corpus/news/sina_inputs.pkl --table_dir ../table/ --train ../corpus/news/sina_train.pkl --model bigram --save ../model/bi_test.pkl --lamb 0 1 | tee ../log/eval_bi_01.log
python -u eval.py --gt ../corpus/news/sina_test.pkl --inputs ../corpus/news/sina_inputs.pkl --table_dir ../table/ --train ../corpus/news/sina_train.pkl --model bigram --load ../model/bi_test.pkl --lamb 0.5 0.5 | tee ../log/eval_bi_55.log
python -u eval.py --gt ../corpus/news/sina_test.pkl --inputs ../corpus/news/sina_inputs.pkl --table_dir ../table/ --train ../corpus/news/sina_train.pkl --model bigram --load ../model/bi_test.pkl --lamb 1 0 | tee ../log/eval_bi_10.log
python -u eval.py --gt ../corpus/news/sina_test.pkl --inputs ../corpus/news/sina_inputs.pkl --table_dir ../table/ --train ../corpus/news/sina_train.pkl --model bigram --load ../model/bi_test.pkl --lamb 0.3 0.7 | tee ../log/eval_bi_37.log