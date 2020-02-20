#include <iostream>
#include "util.h"

// no a priori reason for any incoming message to be spam rather than ham,
// and thus this classifier considers both cases to have equal probabilities
#define SPAM_PRIOR 0.5
#define HAM_PRIOR (1 - SPAM_PRIOR)

// number of spam and ham emails encountered in the training dataset
size_t num_spam_emails = 0;
size_t num_ham_emails = 0;

/**** function prototypes ****/
ProbDictPair learn_distributions(const FileListPair&);
Classification classify_new_email(const FilePath&, const ProbDictPair&,
    double zeta = 1.0, const ProbPair& prior_by_category = {SPAM_PRIOR, HAM_PRIOR});
Prob prob_class_intrsct_words(const ProbDict&, const FreqDict&, const Prob&, const EmailClass&);
void evaluate_filter_performance(const DirPath&, const ProbDictPair&,
    double zeta = 1.0, const ProbPair& prior_by_category = {SPAM_PRIOR, HAM_PRIOR});

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
    // get word frequency in spam and ham emails in the training dataset [w_i] --> [f_i]
    FreqDict spam_freq = get_word_freq_in_files(file_lists_by_category[0]);
    FreqDict ham_freq = get_word_freq_in_files(file_lists_by_category[1]);

    // get number of spam and ham emails in the training dataset
    num_spam_emails = file_lists_by_category[0].size();
    num_ham_emails = file_lists_by_category[1].size();

    // create a smoothed estimate of the probabilities P(w_i|SPAM) and P(w_i|HAM)
    // P(w_i|SPAM/HAM) = ((f_i|SPAM/HAM) + 1) / (#(SPAM/HAM) + 2)
    ProbDict spam_prob;
    for (const auto& word : spam_freq)
        spam_prob[word.first] = (Prob) (word.second + 1)/ (Prob) (num_spam_emails + 2);

    ProbDict ham_prob;
    for (const auto& word : ham_freq)
        ham_prob[word.first] = (Prob) (word.second + 1)/ (Prob) (num_ham_emails + 2);

    return {spam_prob, ham_prob};
}

/**
 * uses naive Bayes classification to classify the email in the given file
 *
 * @param email_path : path of the file to be classified
 * @param probabilities_by_category : output of the learn_distributions() function
 * @param zeta : decision factor; if [ln P(SPAM|Email)] > zeta * [ln P(HAM|Email)],
 *  then the email will be classified as SPAM, and HAM otherwise (empirically optimized).
 * @param prior_by_category : A two-element array as prior probability distribution
 *  for SPAM and HAM email classes
 * @return classification result (std::pair<EmailClass, Prob>) for the given email.
 *  the first element is of type EmailClass (SPAM or HAM) and the second element is a
 *  two-element array as [ln P(SPAM|Email), ln P(HAM|Email)], representing the natural log of
 *  posterior probabilities
 */
Classification classify_new_email(const FilePath& email_path, const ProbDictPair& probabilities_by_category,
    double zeta, const ProbPair& prior_by_category)
{
    Classification classify_result;

    // get frequency of words in email
    FreqDict word_freq = get_word_freq_in_file(email_path);

    // calculate probability of spam and ham intersect with words in the email
    Prob spam_intrsct_words = prob_class_intrsct_words(probabilities_by_category[0], word_freq,
            prior_by_category[0], EmailClass::SPAM);
    Prob ham_intrsct_words = prob_class_intrsct_words(probabilities_by_category[1], word_freq,
            prior_by_category[1], EmailClass::HAM);

    // decide email class
    if (spam_intrsct_words > zeta*ham_intrsct_words)
        classify_result.first = EmailClass::SPAM;
    else
        classify_result.first = EmailClass::HAM;

    // calculate [ln P(SPAM|Email)] and [ln P(HAM|Email)]
    Prob spam_given_email = spam_intrsct_words - log(exp(spam_intrsct_words) + exp(ham_intrsct_words));
    Prob ham_given_email = ham_intrsct_words - log(exp(spam_intrsct_words) + exp(ham_intrsct_words));

    classify_result.second = {spam_given_email, ham_given_email};
    return classify_result;
}

/**
 * calculates [ln P(Email and Class)]; Email = W = {w_1, ..., w_n}, where w_i is
 * a word in the email; Class corresponds to EmailClass = either SPAM or HAM;
 * Note that, P(Email and Class) = P(Class)*P(Email|Class);
 *
 * @param word_class_prob : dictionary whose keys are email words and values are P(w_i|Class)
 * @param word_email_freq : dictionary whose keys are email words and values are f_(w_i)
 * @param class_prior_prob : prior probability of the email class
 * @return probability of class intersect words of the email, that is, probability of both the
 *  words and class appearing or taking place
 */
Prob prob_class_intrsct_words(const ProbDict& word_class_prob, const FreqDict& word_email_freq,
    const Prob& class_prior_prob, const EmailClass& email_class)
{
    // P(Class ⋂ Words) = P(Class) * P (Words|Class), where
    // P(Words|Class) = (\sum w_i)!/(\prod w_i!) * (\prod P(w_i|Class)^f_(w_i))

    // initialize intersection probability with prior class probability
    Prob prob_cls_int_wrd = log(class_prior_prob);

    // initialize numerator and denominator of the multinomial term
    long double num = 0.0;
    long double den = 1.0;
    Prob prob_word_given_class = 0.0; // ln()

    // calculate [ln P(Words|Class)] incrementally
    for (const auto& word : word_email_freq)
    {
        // if word not seen before, update probability with a non-zero smoothed estimate
        if (word_class_prob.find(word.first) == word_class_prob.end())
        {
            prob_word_given_class += (email_class == EmailClass::SPAM) ?
                    log((Prob) 1/ (Prob) (num_spam_emails + 2)) : log((Prob) 1/ (Prob) (num_ham_emails + 2));
            num += 1;
            den = den; // den += log(1)
        }
        else
        {
            prob_word_given_class += (word.second)*log(word_class_prob.at(word.first));
            num += word.second;
            den += lgamma(word.second + 1.0);
        }
    }

    // update intersection probability
    prob_cls_int_wrd += lgamma(num + 1.0) - den;
    prob_cls_int_wrd += prob_word_given_class;

    return prob_cls_int_wrd;
}

/**
 * tests filter performance over the given email files
 *
 * @param test_dir : path to directory holding all test emails to be classified
 * @param probabilities_by_category : output of the learn_distributions() function
 * @param zeta : decision factor; if [ln P(SPAM|Email)] > zeta * [ln P(HAM|Email)],
 *  then the email will be classified as SPAM, and HAM otherwise (empirically optimized).
 * @param prior_by_category : A two-element array as prior probability distribution
 *  for SPAM and HAM email classes
 */
void evaluate_filter_performance(const DirPath& test_dir, const ProbDictPair& probabilities_by_category,
    double zeta, const ProbPair& prior_by_category)
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
        Classification classify_result = classify_new_email(email, probabilities_by_category,
                                                            zeta, prior_by_category);

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
//  errors can be traded off, and plot the trade-off curve (modify zeta)
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