#include <iostream>
#include "util.h"

// no a priori reason for any incoming message to be spam rather than ham,
// and thus this classifier considers both cases to have equal probabilities
#define SPAM_PRIOR 0.5
#define HAM_PRIOR (1 - SPAM_PRIOR)

/**** function prototypes ****/
ProbDictPair learn_distributions(const FileListPair&);
Classification classify_new_email(const FilePath&, const ProbDictPair&,
    const ProbPair& prior_by_category = {SPAM_PRIOR, HAM_PRIOR});
void evaluate_filter_performance(const DirPath&, const ProbDictPair&,
     const ProbPair& prior_by_category = {SPAM_PRIOR, HAM_PRIOR});

/**** functions ****/

/**
 * estimates parameters P(w_i|SPAM) and P(w_i|HAM) for all w_i from the training set
 *
 * @param file_lists_by_category : a two-element array. the first element is a list of
 *  spam files and the second element is a list of ham files
 * @return probabilities_by_category : a two-element array. the first element is a dictionary
 *  whose keys are words and values are smoothed estimates of P(w_i|SPAM); the second element
 *  is a dictionary whose keys are words and values are smoothed estimates of P(w_i|HAM)
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
 *  for SPAM and HAM email classes
 * @return classification result (std::pair<EmailClass, Prob>) for the given email.
 *  the first element is of type EmailClass (SPAM or HAM) and the second element is a
 *  two-element array as [log P(SPAM|Email), log P(HAM|Email)], representing the logarithms of
 *  posterior probabilities
 */
Classification classify_new_email(const FilePath& email_path, const ProbDictPair& probabilities_by_category,
    const ProbPair& prior_by_category)
{
    Classification classify_result;
    return classify_result;
}

/**
 * tests filter performance over the given email files
 *
 * @param test_dir : path to directory holding all test emails to be classified
 * @param probabilities_by_category : output of the learn_distributions() function
 * @param prior_by_category : A two-element array as prior probability distribution
 *  for SPAM and HAM email classes
 */
void evaluate_filter_performance(const DirPath& test_dir, const ProbDictPair& probabilities_by_category,
     const ProbPair& prior_by_category)
{
    // performance evaluation matrix:
    // [ #(SPAM|SPAM)   ;   #(HAM|SPAM)
    //   #(SPAM|HAM)    ;   #(HAM|HAM) ], where
    // #(SPAM|SPAM) is the number of emails which belong to SPAM class and were classified as SPAM
    // #(HAM|SPAM) is the number of emails which belong to SPAM class and were classified as HAM
    // #(SPAM|HAM) is the number of emails which belong to HAM class and were classified as SPAM
    // #(HAM|HAM) is the number of emails which belong to HAM class and were classified as HAM
    PerformanceMatrix perf_mat = Eigen::Matrix2i::Zero();

    // classify emails from test_dir and measure performance
    FileList test_files = get_files_in_folder(test_dir);
    for (const FilePath& email : test_files)
    {
        // classify email
        Classification classify_result = classify_new_email(email, probabilities_by_category);

        // populate performance matrix based on if classification result correctly
        // matches the email's label
        int true_idx = get_email_label(email);      // SPAM = 0, HAM = 1
        int classify_idx = classify_result.first;   // SPAM = 0, HAM = 1
        perf_mat(true_idx, classify_idx) += 1;
    }

    // get total number of spam and ham emails
    int total_spam = perf_mat(0,0) + perf_mat(0,1);
    int total_ham = perf_mat(1,0) + perf_mat(1,1);

    // print result
    std::cout << "Correctly classified " << perf_mat.diagonal()(0) << " out of "
        << total_spam << " spam emails, and " << perf_mat.diagonal()(1) << " out of "
        << total_ham << " ham emails" << std::endl;
}

// TODO: Write your code here to modify the decision rule such that
//  errors can be traded off, and plot the trade-off curve
// TODO: Pre-train bayesian filter and store in pretrain.txt to make classification faster.

/**** main ****/
int main()
{
    // folders for training and testing
    DirPath spam_dir = "../data/spam/";
    DirPath ham_dir = "../data/ham/";
    DirPath test_dir = "../data/testing";

    // file lists for training
    FileList spam_files = get_files_in_folder(spam_dir);
    FileList ham_files = get_files_in_folder(ham_dir);
    FileListPair training_files = {spam_files, ham_files};

    // learn distributions from training data
    ProbDictPair probabilities_by_category = learn_distributions(training_files);

    // classify test emails and evaluate performance
    evaluate_filter_performance(test_dir, probabilities_by_category);

    return 0;
}