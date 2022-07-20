from functools import reduce
import math
import email
import os
from collections import Counter
import json

TRAIN_HAM = "/data/cs65-S22/hamspam/train-ham/"
TRAIN_SPAM = "/data/cs65-S22/hamspam/train-spam/"
DEV_HAM = "/data/cs65-S22/hamspam/dev-ham/"
DEV_SPAM = "/data/cs65-S22/hamspam/dev-spam/"


def getPaths(dir):
    """Creates a list of the file names from a directory

    Args:
        dir (string) : the directory

    Returns:
        ([]) : a list of file names.
    """
    return [f"{dir}{name}" for name in os.listdir(dir)]


def load_tokens(path):
    """Tokenizes the string body from an email.

    Args:
        path (string) : a filename which contains the email

    Returns:
        ([]):
        An array of subarrays, where each subarray contains each line's non-whitespace and
        space-separated tokens.
    """
    tokens = []
    with open(path, "rb") as file:
        message = email.message_from_binary_file(file)
        for l in email.iterators.body_line_iterator(message):
            for token in l.split(" "):
                if not str.isspace(token) and not token == "":
                    tokens.append(token)
    return tokens


def log_probs(email_paths, smoothing):
    """Builds a log probability dictionary for all types contained in all the emails
    using the Naive Bayes model.

    Args:
        email_paths (array): A list of paths, where each path has an email
        smoothing (float): the smoothing value to apply for the Naive Bayes.

    Returns:
        ({}): A dictionary where key is the type and the value is the probability.
    """
    tokens_list = [load_tokens(path) for path in email_paths]
    all_words = reduce(lambda a, b: a + b, tokens_list)
    frequency = Counter(all_words)
    frequency["<UNK>"] = 0
    V = list(frequency.keys())
    denom = len(all_words) + smoothing * len(V)
    probs = {}

    for w in V:
        probability = math.log((frequency[w] + smoothing) / denom)
        probs[w] = probability
    return probs


# with open("ham.json", "w") as f:
#     json.dump(log_probs(getPaths(train_ham), 10**-5), f)
# with open("spam.json", "w") as f:
#     json.dump(log_probs(getPaths(train_spam), 10**-5), f)

ham_file = open("ham.json")
spam_file = open("spam.json")
# print(json.load(ham_file))


class SpamFilter:
    def __init__(self, spam_dir, ham_dir, smoothing):
        """Initializes the necessary data structure for the Naive Bayes

        Args:
            spam_dir (string): a directory of spam email files
            ham_dir (string): a directory of ham email files
            smoothing (float): the smoothing value to apply for the Naive Bayes
        """
        self.spam_probs = log_probs(getPaths(spam_dir), smoothing)
        self.ham_probs = log_probs(getPaths(ham_dir), smoothing)
        self.spam_class_prob = len(getPaths(spam_dir)) / (
            len(getPaths(ham_dir)) + len(getPaths(spam_dir))
        )
        self.ham_class_prob = 1 - self.spam_class_prob
        self.prob_map = {
            "ham": self.ham_probs,
            "spam": self.spam_probs,
            "ham_class_prob": self.ham_class_prob,
            "spam_class_prob": self.spam_class_prob,
        }

    def most_indicative(self, n, class_type):
        """Computes the n
        most indicative words for the given class type, sorted in descending order
        based on their indication values.

        Args:
            n (int): the number of words returned should not exceed this.
            class_type (string): the category for which to compute the words

        Returns:
            []: a list of tuples, each tuple's first value is the word and the second is the
                corresponding indicative value
        """
        commons = list(set(self.ham_probs) & set(self.spam_probs))
        indicative = {}
        for w in commons:
            value = self.prob_map[class_type][w] - math.log(
                (math.e ** self.spam_probs[w] + math.e ** self.ham_probs[w])
            )
            indicative[w] = value
        indicative = sorted(
            indicative.items(), key=lambda x: x[1], reverse=True
        )
        return [w for w, v in indicative][:n]

    def most_indicative_spam(self, n):
        """Computes most indicative words for the spam class

        Args:
             n (int): the number of words returned should not exceed this.

         Returns:
             []: a list of tuples, each tuple's first value is the word and the second is the
                 corresponding indicative value
        """
        return self.most_indicative(n, "spam")

    def most_indicative_ham(self, n):
        """Computes most indicative words for the ham class

        Args:
             n (int): the number of words returned should not exceed this.

         Returns:
             []: a list of tuples, each tuple's first value is the word and the second is the
                 corresponding indicative value
        """
        return self.most_indicative(n, "ham")

    def class_prob(self, class_type, freqs):
        """Given a class type and a frequency dictionary, this method
        calculates the likelihood of this frequency using the Naive Bayes model

        Args:
            class_type (string): a category, ham or spam in this case
            freqs (_type_): A word  and count frequency dictionary

        Returns:
            (float): the likelihood
        """
        probs = self.prob_map[class_type]
        class_prob = self.prob_map[f"{class_type}_class_prob"]
        likelihood = 0
        for w, count in freqs.items():
            if w not in probs:
                w = "<UNK>"
            # print(probs[w], count)
            likelihood += (probs[w]) * count
        return class_prob * likelihood

    def is_spam(self, email_path):
        """Given an email, determines if it's a ham or a spam

        Args:
            email_path (string): the path which contains the email

        Returns:
            (Boolean): true if spam, false otherwise
        """
        tokens = load_tokens(email_path)
        frequency = Counter(tokens)
        ham_prob = self.class_prob("ham", frequency)
        spam_prob = self.class_prob("spam", frequency)
        return spam_prob > ham_prob


def main():
    paths = getPaths(TRAIN_HAM)
    print(load_tokens(paths[0])[2])
    paths = ["/data/cs65-S22/hamspam/train-ham/ham%d" % i for i in range(1, 11)]
    print(load_tokens(TRAIN_HAM + "ham1")[200:204])
    print(load_tokens(TRAIN_HAM + "ham2")[110:114])
    p = log_probs(paths, 10**-5)
    print(p["the"])
    print(p["line"])
    print(p["<UNK>"])

    # Class
    emali_path = "/data/cs65-S22/hamspam/dev-ham/dev3"
    filter = SpamFilter(TRAIN_SPAM, TRAIN_HAM, 10**-5)
    print(filter.most_indicative_spam(5))
    print(filter.most_indicative_ham(5))
    print(filter.is_spam(emali_path))
    # pass


if __name__ == "__main__":
    main()
