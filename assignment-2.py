import sys, glob, math, re
from nltk import sent_tokenize, word_tokenize, ngrams, FreqDist

'''
Helper function to replace all values in dictionary d below a certain threshold count with unknown token '<UNK>'
'''
def add_unknown_tok(d):
	token = '<UNK>'
	thresh = min(d.values())

	new_d = {}
	for k,v in d.items():
		if v <= thresh:
			if token not in new_d:
				new_d[token] = v
			else:
				new_d[token] += v
		else:
			new_d[k] = v
	return new_d

'''
Helper function that takes in a list of n_grams and generates a count dictionary as our model
'''
def create_count_dictionary(n, n_grams, laplace=False, interpolation=False):
	d = {}	# dictionary
	c = 0	# total ngram counts
	for ngram in n_grams:
		ng = ''.join(ngram)
		if ng not in d:
			c += 1
			d[ng] = 1 if not laplace else 2 # extra count for laplace smoothing
		else:
			c += 1
			d[ng] += 1
	d['ngram_count_total'] = c if not laplace else (c+len(d)) # adjusting total counts for laplace smoothing

	if laplace:
		return d
	elif interpolation:
		unigram_count = 0
		for k,v in d.items():
			if len(k) == 1:
				unigram_count += v

		d['unigram_count'] = unigram_count
		return add_unknown_tok(d)

	else:
		return add_unknown_tok(d)

'''
Given training text and parameter n, returns a list of lamda values computed as described in the deleted interpolation
algorithm, as described in the J&M textbook (​chapter 8, pages 15-16, 3rd edition​)
'''
def del_interpolation(model, n):
	lambda_vals = [0]*n
	m = {}
	for k,v in model.items():
		if len(k) == n:
			max_case_val = -1
			max_case_val_index = -1
			for i in range(0, n):
				case_no = n - i - 1
				ng = k[i:]
				numerator = (model['<UNK>'] - 1) if ng not in model else (model[ng] - 1)
				if len(ng) == 1:
					case = numerator / (model['unigram_count'] - 1)
				else:
					context = ng[1:]
					denom = (model['<UNK>'] - 1) if context not in model else (model[context] - 1)
					case = 0 if denom == 0 else numerator / denom

				if case > max_case_val:
					max_case_val = case
					max_case_val_index = case_no

			lambda_vals[max_case_val_index] += v

	# Normalizing lambda vals
	val_sum = sum(lambda_vals)
	lambda_vals_norm = [lambda_val / val_sum for lambda_val in lambda_vals]

	return lambda_vals_norm

'''
Takes parameters n, text, computes the n and n-1 grams of the text and returns their frequency distribution
dictionaries as a tuple in case of unsmoothed and laplace models. If interpolation in specified, it instead returns
a freq distribution dictionary for all n, n-1, ..., 1 grams, as well as a list of the computed lambda parameters
'''
def create_ngrams(n, text, laplace=False, interpolation=False):
	txt = ' '.join(text.split()) # Remove all tabs, newlines, whitespaces with a single whitespace
	txt = '_' + txt.replace(' ', '_') + '_'

	if interpolation:
		all_ngrams = []
		for i in range(1, n+1):
			n_grams = ngrams(txt, i)
			for ng in n_grams:
				all_ngrams.append(ng)
		
		m = create_count_dictionary(n, all_ngrams, False, interpolation)
		lambda_vals = del_interpolation(m, n)
		#print(lambda_vals)
		return (m, lambda_vals)

	n_grams = (ngrams(txt, n), ngrams(txt, n-1))

	m1 = create_count_dictionary(n, n_grams[0], laplace)
	m2 = create_count_dictionary(n, n_grams[1], laplace)

	return (m1, m2)
'''
This function takes in a parameters n (the value for the ngram), model (tuple of dictionaries, or dictionary and a list of
lambda values), the test text, and it computes the probabilities for ngrams in this test text, returning the perplexity score
'''
def compute_pp(n, model, txt, laplace=False, interpolation=False):
	t = ' '.join(txt.split())# Clean up the test_txt
	t = '_' + t.replace(' ', '_') + '_'

	n_grams = ngrams(t, n)

	N = 0 # Total number of ngrams, for perplexity computation
	logprob = 0
	for ngram in n_grams:
		ng = ''.join(ngram)

		if interpolation:
			lambda_vals = model[1]
			prob = 0 # combined probability for ngram with lambda coefficients
			for i in range(0, n):

				lambda_i = lambda_vals[i]

				n_gram = ng[i:]

				if n_gram not in model[0]:
					if len(n_gram) == 1:
						prob += lambda_i*(model[0]['<UNK>'] / model[0]['unigram_count'])
				else:
					context = n_gram[1:]

					num = model[0][n_gram]
					if len(n_gram) == 1:
						denom = model[0]['unigram_count']
					else:
						denom = model[0]['<UNK>'] if context not in model[0] else model[0][context]

					prob += lambda_i*(num / denom)

			if prob <= 0:
				print("HEY")
				continue
			else:
				logprob += math.log(prob)

		else:
			context = ng[:n-1] # Previous n-1 chars

			if ng not in model[0]:
				if laplace:
					logprob += math.log(1 / (len(model[1]) - 1))
				else:
					logprob += math.log(model[0]['<UNK>'] / model[0]['ngram_count_total'])
			else:
				ng_count = model[0][ng]
				context_count = model[1][context] if context else model[0]['ngram_count_total']
				logprob += math.log(ng_count / context_count)

		N += 1

	pp = math.exp(((logprob / N)*-1))

	return pp

'''
Helper function that takes our results list and returns the percentage of accurately classified text files 
'''
def eval_results(results):
	total = len(results)
	wrong = 0
	for result in results:
		a = result[0].split('.')[0]
		b = result[1].split('.')[0]

		if a != b:
			wrong += 1
			#print("File: " + a + "	Classified: " + b)

	acc = (total - wrong)/total
	return round(acc*100, 2)

'''
Helper function to generate tab separated text output. Requited parameters are filenam and list of results
'''
def generate_output_file(filename, results):
	filename += '.txt'
	with open(filename, 'w') as f:
		for result in results:
			f.write('\t'.join(result[0:]) + '\n')

	print("Output generated successfully! Please check your current working directory for the output .txt file.")

def main():
	if len(sys.argv) <= 1 or sys.argv[1] not in ["--unsmoothed", "--laplace", "--interpolation"]:
		print('Program requires one argument indicating the type of smoothing: "--unsmoothed", "--laplace" or "--interpolation"')
		return

	laplace = False
	interpolation = False
	train_type = sys.argv[1]
	out_filename = 'results_'

	test_type = input('Are you testing for test or dev files? Enter t for test, d for dev.\n')
	if test_type.lower() == "d":
		glob_name = '*_dev/*'
		out_filename += 'dev_'
	elif test_type.lower() == "t":
		glob_name = '*_test/*'
		out_filename += 'test_'
	else:
		print('Please specify correct option (either t or d) and try again.')
		return

	# Selecting tuned value of n
	if train_type == '--unsmoothed':
		n = 1
		out_filename += 'unsmoothed'
	elif train_type == '--laplace':
		laplace = True
		n = 7
		out_filename += 'add-one'
	else:
		interpolation = True
		n = 4
		out_filename += 'interpolation'

	models = {}
	results = []

	### TRAINING ###
	for filepath in glob.glob('*_train/*'):
		match = re.search(r"/.+", filepath)
		if match is not None:
			model_name = match.group(0)[1:]
			
			file = open(filepath, 'r')
			txt = file.read()
			file.close()

			model = create_ngrams(n, txt, laplace, interpolation)
			models[model_name] = model

		else:
			print('Please make sure you have files in the directory.')
			return

	### TESTING ###
	for filepath in glob.glob(glob_name):
		match = re.search(r"/.+", filepath)
		if match is not None:
			filename = match.group(0)[1:]

			test_file = open(filepath, 'r')
			test_txt = test_file.read()
			test_file.close()

			model_choice = ''
			pp_score = float('inf')

			for model_name, model in models.items():
				pp = compute_pp(n, model, test_txt, laplace, interpolation)
				if pp < pp_score:
					model_choice = model_name
					pp_score = pp

			result = [filename, model_choice, str(round(pp_score, 2))]
			results.append(result)

		else:
			print('Please make sure you have files in the directory.')
			return

	#print(eval_results(results))
	generate_output_file(out_filename, results)

main()