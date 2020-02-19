#include <iostream>
#include <array>
#include "util.h"

// no a priori reason for any incoming message to be spam rather than ham,
// and thus this classifier considers both cases to have equal probabilities
#define SPAM_PRIOR 0.5
#define HAM_PRIOR (1 - SPAM_PRIOR)

/**** function prototypes ****/
ProbDictPair learn_distributions(const FileListPair&);
Classification classify_new_email(const FilePath&, const ProbDictPair&,
        const ProbPair& prior_by_category = {SPAM_PRIOR, HAM_PRIOR});

/**** functions ****/

/**
 * estimates parameters P(w_i|SPAM) and P(w_i|HAM) for all w_i from the training set
 *
 * @param file_lists_by_category : a two-element array. the first element is a list of
 * spam files and the second element is a list of ham files
 * @return probabilities_by_category : a two-element array. the first element is a dictionary
 * whose keys are words and values are smoothed estimates of P(w_i|SPAM); the second element
 * is a dictionary whose keys are words and values are smoothed estimates of P(w_i|HAM)
 */
ProbDictPair learn_distributions(const FileListPair& file_lists_by_category)
{
    ProbDictPair probabilities_by_category;
    return probabilities_by_category;
}

/**
 * uses naive Bayes classification to classify the email in the given file
 *
 * @param email_path : path of the file to be classified
 * @param probabilities_by_category : output of the learn_distributions() function
 * @param prior_by_category : A two-element array as prior probability distribution
 * for SPAM and HAM email classes
 * @return classification result (std::pair<EmailClass, Prob>) for the given email.
 * the first element is of type EmailClass (SPAM or HAM) and the second element is a
 * two-element array as [log P(SPAM|Email), log P(HAM|Email)], representing the logarithms of
 * posterior probabilities
 */
Classification classify_new_email(const FilePath& email_path, const ProbDictPair& probabilities_by_category,
        const ProbPair& prior_by_category)
{
    Classification classify_result;
    return classify_result;
}

/**** main ****/
int main()
{
    FileList files = get_files_in_folder("../data/ham/");
    FreqDict words_freq = get_word_freq(files);
    for (const auto& it : words_freq)
        std::cout << it.first << " : " << it.second << std::endl;
    return 0;
}