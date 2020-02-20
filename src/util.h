#ifndef CLASSIFIER_UTIL_H
#define CLASSIFIER_UTIL_H

#include <fstream>
#include <string>
#include <array>
#include <unordered_map>
#include <eigen3/Eigen/Eigen>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/filesystem.hpp>
#include "matplotlib.h"

/**** type definitions ****/
enum EmailClass {SPAM = 0, HAM = 1};
typedef long double Prob;                                   // probability
typedef std::string DirPath;                                // folder path
typedef std::string FilePath;                               // file path
typedef std::vector<FilePath> FileList;                     // list of file paths
typedef std::vector<std::string> WordList;                  // list of words (with repetition and unsorted)
typedef std::unordered_map<std::string, Prob> ProbDict;     // dictionary of probabilities
typedef std::unordered_map<std::string, size_t> FreqDict;   // dictionary of frequencies
typedef std::array<FileList, 2> FileListPair;               // two-element array of file lists
typedef std::array<ProbDict, 2> ProbDictPair;               // two-element array of probability dictionaries
typedef std::array<Prob, 2> ProbPair;                       // two-element array of probabilities
typedef std::pair<EmailClass, ProbPair> Classification;     // a pair of email class and two-element array of probabilities
typedef Eigen::Matrix2i PerformanceMatrix;                  // 2x2 matrix containing number of emails classified as:
                                                            // [ #(SPAM|SPAM)   ;   #(HAM|SPAM)
                                                            //   #(SPAM|HAM)    ;   #(HAM|HAM) ]

namespace plt = matplotlibcpp;
namespace fs = boost::filesystem;
namespace bsm = boost::math;

extern size_t num_spam_emails;
extern size_t num_ham_emails;

/**** function prototypes ****/
FileList get_files_in_folder(const DirPath&, const std::string& extension = ".txt");
WordList get_words_in_file(const FilePath&);
FreqDict get_word_freq_in_files(const FileList&);
FreqDict get_word_freq_in_file(const FilePath&);
EmailClass get_email_label(const FilePath&);
void plot_probabilities(const ProbDict&);

/**** functions ****/
FileList get_files_in_folder(const DirPath& dir_path, const std::string& extension)
{
    FileList file_list;
    fs::path path = fs::system_complete(dir_path);

    if (!path.empty())
    {
        fs::directory_iterator end;

        for (auto i = fs::directory_iterator(path); i != end; ++i)
        {
            if (fs::extension(i->path()) == extension)
                file_list.push_back(i->path().string());
        }
    }

    return file_list;
}

WordList get_words_in_file(const FilePath& file_path)
{
    WordList word_list;
    std::ifstream file(file_path);

    std::string word;
    while (file >> word)
        word_list.push_back(word);

    return word_list;
}

FreqDict get_word_freq_in_files(const FileList& files)
{
    FreqDict freq_dict;

    for (const FilePath& file : files)
    {
        WordList words = get_words_in_file(file);
        for (const std::string& word : words)
            if (!word.empty())
                ++freq_dict[word];
    }

    return freq_dict;
}

FreqDict get_word_freq_in_file(const FilePath& file_path)
{
    FreqDict freq_dict;

    WordList words = get_words_in_file(file_path);
    for (const std::string& word : words)
        if (!word.empty())
            ++freq_dict[word];

    return freq_dict;
}

EmailClass get_email_label(const FilePath& email_path)
{
    std::string file_name = fs::path(email_path).filename().string();

    if (file_name.find("spam") != std::string::npos)
       return EmailClass::SPAM;
    return EmailClass::HAM;
}

void plot_probabilities(const ProbDict& prob_dict)
{
//    std::vector<double> x, y;
//    size_t lbl_cnt = 1;
//
//    // plot all map points
//    for (const Point& p : pnt_arr)
//    {
//        x.push_back(p[0]);
//        y.push_back(p[1]);
//    }
//    plt::scatter(x, y, 3);
//
//    // plot k nearest points and annotate
//    x.clear();
//    y.clear();
//    for (const Point& p : k_arr)
//    {
//        x.push_back(p[0]);
//        y.push_back(p[1]);
//        plt::annotate(std::to_string(lbl_cnt), p[0], p[1]);
//        lbl_cnt++;
//    }
//    plt::scatter(x, y, 15);
//
//    // plot input point and annotate
//    plt::annotate("IN", usr_pnt[0], usr_pnt[1]);
//    plt::scatter(std::vector<double>{usr_pnt[0]}, std::vector<double>{usr_pnt[1]}, 20);
//
//    plt::title("kNN Search Result");
//    plt::show();
}

#endif //CLASSIFIER_UTIL_H
