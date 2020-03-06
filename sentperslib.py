import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from scipy.spatial import distance
import sys
import codecs

bins = 32
frequency = 3000
trait_name = ["Openness", "Conscentiousness", "Extraversion",
              "Agreableness", "Neuroticism"]
gloveFilePath = "../dataset/glove.6B.300d"
schwartzNames = ["selfdirection", "stimulation", "hedonism", "achievement",
                 "power", "security", "conformity", "tradition",
                 "benevolence", "universalism"]
schwartzCentroidsFilePath = "bhv_centroids.csv"


def kl_divergence(p, q):
    return (p*np.log(p/q)).sum()


def dist_info(trait_arr, trait_name):
    print(trait_name)
    print("mean, std, var", np.mean(np.asarray(trait_arr)),
          np.std(np.asarray(trait_arr)), np.var(np.asarray(trait_arr)))


def read_and_display_distribution(input_csv, title):
    X = pd.read_csv(input_csv, header=None)
    o = X.iloc[0, :]
    c = X.iloc[1, :]
    e = X.iloc[2, :]
    a = X.iloc[3, :]
    n = X.iloc[4, :]
    pos = 0
    ocean = [o, c, e, a, n]
    for trait in ocean:
        dist_info(trait, trait_name[pos])
        plt.title(title+" "+str(trait_name[pos]))
        plt.hist(trait, bins=bins, range=(1, 5))
        plt.ylabel("frequency")
        plt.xlabel("score")
        plt.axis([1, 5, 0, frequency])
        plt.grid()
        plt.draw()

        plt.savefig("./img/"+title+"_"+str(trait_name[pos])+'.png', dpi=100)

        plt.close()
        pos = pos + 1

    return ocean


def mean_big5_768(dataset_path):
    '''
        Compute and return the mean of O,C,E,A,N of dataset.
        Dataset must be a csv with no header and
        768 feature from embedding phase +
        5 traits in the ocean order.
    '''
    # original dataset "../train_whole_lines.csv"
    dataset = pd.read_csv(dataset_path, header=None)
    Y = dataset.iloc[:, 768:]  # Big5 scores related to X
    o = Y.iloc[:, 0]
    c = Y.iloc[:, 1]
    e = Y.iloc[:, 2]
    a = Y.iloc[:, 3]
    n = Y.iloc[:, 4]
    return np.mean(o), np.mean(c), np.mean(e), np.mean(a), np.mean(n)


def mean_big5_5lines(dataset_path):
    '''
        Open a csv dataset made up of 5 lines (o,c,e,a,n)
        Return the 5 means
    '''
    big5_mean = []
    fi = open(dataset_path, "r")
    for i in range(5):
        line = fi.readline()
        line = line.rstrip("\n")
        elements = line.split(",")
        mean = np.array(elements, dtype=float)
        mean = np.mean(mean, axis=0)
        big5_mean.append(mean)
    fi.close()
    return big5_mean


def radar_plot(handle, big5, filename, big5_mean):
    # Number of variables
    categories = ['OPE', 'CON', 'EXT', 'AGR', 'NEU']
    N = len(categories)

    # What will be the angle of each axis in the plot?
    # (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4], ['1', '2', '3', '4'], color="grey", size=7)
    plt.ylim(0, 5)

    # Ind1
    values = big5
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values = big5_mean
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Mean")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add title
    plt.suptitle("Submitted text Big5", size=16)
    plt.subplots_adjust(top=0.83)

    # Save file
    plt.savefig("flaskr/static/images/"+filename+".jpg")
    plt.close()


def compute_centroids(filename):
    fout = open(filename, "w")
    # file name of the output file created in the same directory
    selfdirection = ["creativity", "freedom", "goal-setting", "curious",
                     "independent", "self-respect", "intelligent", "privacy"]
    stimulation = ["excitement", "novelty", "challenge", "variety",
                   "stimulation", "daring"]
    hedonism = ["pleasure", "sensuous",  "gratification", "enjoyable",
                "self-indulgent"]
    achievement = ["ambitious", "successful", "capable", "influential",
                   "intelligent", "self-respect"]
    power = ["authority", "wealth", "power", "reputation", "notoriety"]
    security = ["safety", "harmony", "stability", "order", "security", "clean",
                "reciprocation", "healthy", "moderate", "belonging"]
    conformity = ["obedient", "self-discipline", "politeness", "honoring",
                  "loyal", "responsible"]
    tradition = ["tradition", "humble", "devout", "moderate", "spiritualist"]
    benevolence = ["helpful", "honest", "forgiving", "responsible", "loyal",
                   "friendship", "love", "meaningful"]
    universalism = ["broadminded", "justice", "equality", "peace", "beauty",
                    "environment-friendly", "wisdom", "environmentalist",
                    "harmony"]
    schwartzBasicHumanValues = [selfdirection, stimulation, hedonism,
                                achievement, power, security, conformity,
                                tradition, benevolence, universalism]

    glove = loadGloveModel(gloveFilePath)
    pos = 0
    schwartzCentroids = {}
    for humanValue in schwartzBasicHumanValues:
        count_elements = 0.0
        schwartzNCentroid = [0.0]
        schwartzNCentroid = schwartzNCentroid*300
        schwartzNCentroid = np.asarray(schwartzNCentroid)
        for representativeWord in humanValue:
            schwartzNCentroid = schwartzNCentroid + \
                                np.asarray(glove[representativeWord])
            count_elements += 1
        schwartzCentroids[schwartzNames[pos]] = \
            schwartzNCentroid/count_elements
        fout.write(','.join(map(str, schwartzCentroids[schwartzNames[pos]])))
        fout.write("\n")
        pos += 1
    fout.close()


def loadGloveModel(gloveFile):
    # "Loading Glove Model"
    # to make it faster
    # https://blog.ekbana.com/loading-glove-pre-trained-word-embedding-model-from-text-file-faster-5d3e8f2b8455
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    f.close()
    return model


def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in
    text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot
    less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') \
            as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))


def load_embeddings_binary(embeddings_path):
    """
    It loads embedding provided by glove which is saved as binary file.
    Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model


def get_w2v(sentence, model):
    """
    :param sentence: inputs a single sentences whose
    word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding
    of all words in input sentence.
    """
    return np.array([model.get(val, np.zeros(100))
                    for val in sentence.split()], dtype=np.float64)


def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def compute_bhv(text):
    final_bhv = []
    # localModel = loadGloveModel(gloveFilePath)
    localModel = load_embeddings_binary(gloveFilePath)
    total_words = {}
    cumulative_vectors = {}
    schwartzCentroids = {}
    fileCentroids = open(schwartzCentroidsFilePath, "r")
    NON_BMP_RE = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]",
                            flags=re.UNICODE)
    for category in schwartzNames:
        total_words[category] = 0
        cumulative_vectors[category] = np.asarray([0.0]*300)
        line = fileCentroids.readline()
        line = line.rstrip("\n")
        elem = line.split(",")
        schwartzCentroids[category] = np.asarray(elem, dtype=float)
    doc_complete = text.split('\n')
    doc_cleaned = [clean(doc).split() for doc in doc_complete]

    for line in doc_cleaned:
        for word in line:
            if word.startswith('@') or word.isdigit() or ("http" in word):
                continue
            else:
                word = NON_BMP_RE.sub('', word)
                if len(word) > 0:
                    if word in localModel:
                        min_distance = sys.float_info.max
                        which_schwartz = ""
                        for pos in schwartzNames:
                            now_distance = \
                                distance.euclidean(np.asarray(localModel.get(word, np.zeros(100)), dtype=float), schwartzCentroids[pos])
                            if now_distance < min_distance:
                                min_distance = now_distance
                                which_schwartz = pos
                        total_words[which_schwartz] += 1
                        cumulative_vectors[which_schwartz] += \
                            np.asarray(localModel[word])
    for category in schwartzNames:
        if total_words[category] != 0:
            now_centroid = cumulative_vectors[category]/total_words[category]
            dist = distance.euclidean(now_centroid, schwartzCentroids[pos])
            if dist != 0:
                final_bhv.append(str(round(total_words[category]*(1/dist), 3)))
            else:
                final_bhv.append("max")
        else:
            final_bhv.append(0)

    return final_bhv
