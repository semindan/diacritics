# Install  
`git clone https://gitlab.mff.cuni.cz/semind/hw1-diacritics-restoration`
# Dependencies  
pytorch  
unidecode
## Environment  
### Create
`conda env create --name semind_hw1 --file=environment.yml`
### Activate
`conda activate semind_hw1`
### Deactivate
`conda deactivate semind_hw1`
### Remove
`conda env remove --name semind_hw1`
# Usage
## Download the data
use `make all` to download the data, train and evaluate on the dev dataset  
## Evaluation and print
use `make eval` to evaluate on the test dataset  
use `make print_dev` to predict the text from the dev dataset and print it to stdout  
use `make print_test` to predict the text from the test dataset and print it to stdout  
## Clean
use `make clean` to erase trained weights and data  
## Text to text script
use `app.py` to read from stdin and print the prediction to stdout
```
python app.py
tento text je nejlepsi
*ctrl-d or ctrl-z (windows) to stop the input*
```
also `cat my_text_file.txt | app.py` or for instance `app.py < my_text_file.txt` work
