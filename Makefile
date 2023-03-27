all:
	mkdir data
	curl -o data/data.tgz http://ctdc.kiv.zcu.cz/czech_text_document_corpus_v20.tgz
	tar -xf data/data.tgz -C data/
	cp data/czech_text_document_corpus_v20/dev/* data/czech_text_document_corpus_v20/
	python extract_corpus.py
	curl -o data/dev.txt https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/diacritics-dtest.txt
	curl -o data/test.txt https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/diacritics-etest.txt
	python main.py --verbose --no_prediction_print --path_train data/text.txt --path_dev data/dev.txt --path_test data/dev.txt

eval:
	python main.py --verbose --no_prediction_print --use_pretrained --path_train data/text.txt --path_dev data/dev.txt --path_test data/test.txt

print_dev:
	python main.py --use_pretrained --path_train data/text.txt  --path_dev data/dev.txt --path_test data/dev.txt

print_test:
	python main.py --use_pretrained --path_train data/text.txt  --path_dev data/dev.txt --path_test data/test.txt

clean:
	rm -r data
	rm *.pt *.pkl

	
