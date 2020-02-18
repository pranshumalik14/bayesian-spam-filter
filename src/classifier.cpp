#include <iostream>
#include <array>
#include "util.h"

#define SPAM_PRIOR 0.5
#define HAM_PRIOR (1 - SPAM_PRIOR)

/**** function prototypes ****/
std::array<double, 2> learn_distributions(std::array<FileList, 2>);
std::pair<CLASS, double> classify_new_email(FilePath, std::array<ProbDict, 2>,
        std::array<double, 2> prior_by_category = {SPAM_PRIOR, HAM_PRIOR});

/**** functions ****/
//def learn_distributions(file_lists_by_category):
//"""
//Estimate the parameters p_d, and q_d from the training set
//
//Input
//-----
//file_lists_by_category: A two-element list. The first element is a list of
//        spam files, and the second element is a list of ham files.
//
//Output
//------
//probabilities_by_category: A two-element tuple. The first element is a dict
//whose keys are words, and whose values are the smoothed estimates of p_d;
//the second element is a dict whose keys are words, and whose values are the
//        smoothed estimates of q_d
//"""
//### TODO: Write your code here
//
//
//return probabilities_by_category
//
//        def classify_new_email(filename,probabilities_by_category,prior_by_category):
//"""
//Use Naive Bayes classification to classify the email in the given file.
//
//Inputs
//------
//filename: name of the file to be classified
//        probabilities_by_category: output of function learn_distributions
//prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the
//        parameter in the prior class distribution
//
//        Output
//------
//classify_result: A two-element tuple. The first element is a string whose value
//is either 'spam' or 'ham' depending on the classification result, and the
//        second element is a two-element list as [log p(y=1|x), log p(y=0|x)],
//representing the log posterior probabilities
//"""
//### TODO: Write your code here
//
//
//return classify_result

/**** main ****/
int main()
{
    FileList files = get_files_in_folder("../data/ham/");
    FreqDict words_freq = get_word_freq(files);
    for (const auto& it : words_freq)
        std::cout << it.first << " : " << it.second << std::endl;
    return 0;
}