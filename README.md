# Operation__HAY
Files:
1) Making_dataset.py
2) train.py
3) inference1.py
4) inference2.py
5) try2.pth

Aim: 1.To classify the expression as infix/prefix/postfix
     2. To evaluate the value of expression

Working Algo:
1. To classify the detected character as appropriate number (0 or 1 or 2 or 3..) or the operator(+ or - or ..)
	Here we used opencv and numpy libraries to split the dataset images into three parts and apply pre-processing(Making_dataset.py)
	to finally train our model using Convoltuional Neural Network (CNN) to classify them as appropriate number or operator.(train.py)
2. To check the location of operator and classify the expression as infix/prefix/postfix and store the output data in csv file. 
	(inference1.py)
3. Evaluating the value of expression and storing the output values in a csv file.
	(inference2.py)
  
  try2.pth is the final model used
