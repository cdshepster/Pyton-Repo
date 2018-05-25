#!/usr/bin/python

import argparse
import cv2
import sys
import os
import cPickle as pickle
from matplotlib import pyplot as plt
from os.path import basename

#######################################################################################################
# module: hist_image_retrieval.py
# author:  Cole Shepherd A01895001
# description: persistent image retriever
# to run:
# $ python hist_image_retrieval.py -ip horfood_test/img01.JPG -hist hsv_hist16.pck -bin 16 -sim bhatta
#
# the output should print the matches for the top 3 images and display the input image
# and the top 3 matches in 4 matplotlib figures.
# images/123472793.JPG --> 0.914982328755
# images/123465243.JPG --> 0.921478476016
# images/123465992.JPG --> 0.923478808005
########################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument('-ip', '--imgpath', required=True, help='image path')
ap.add_argument('-hist', '--hist', required=True, help='hist index file')
ap.add_argument('-bin', '--bin', required=True, help='hist bin size')
ap.add_argument('-sim', '--sim', required=True, help='hist similarity')
ap.add_argument('-clr', '--clr', required=True, help='color space')
args = vars(ap.parse_args())

inimg = cv2.imread(args['imgpath'])
bin_size = int(args['bin'])
# compute the histogram of inimg and save it in inhist
if args['clr'] == 'rgb':
    inhist = cv2.calcHist([inimg], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 256, 0, 256, 0, 256])
elif args['clr'] == 'hsv':
    inhist = cv2.calcHist([inimg], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 180, 0, 256, 0, 256])
# normalize and flatten the inhist into a feature vector
inhist_vec = cv2.normalize(inhist, inhist).flatten()

# get the similarity metric string from the command line parameter.
hist_sim = args['sim']

HIST_INDEX = None

RatingDictionary = {}


def hist_correl_sim(norm_hist1, norm_hist2):
    return cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_CORREL)


def hist_chisqr_sim(norm_hist1, norm_hist2):
    return cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_CHISQR)


def hist_intersect_sim(norm_hist1, norm_hist2):
    return cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_INTERSECT)


def hist_bhatta_sim(norm_hist1, norm_hist2):
    return cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_BHATTACHARYYA)


# compute the topn matches using the value saved in hist_sim above.
def compute_hist_sim(inhist_vec, hist_index, topn=3):
    if hist_sim == 'correl':
        for item in hist_index:
            score = hist_correl_sim(inhist_vec, hist_index.get(item))
            RatingDictionary[item] = score
        topMatches = find_biggest_matches(RatingDictionary, topn)
        return topMatches

    elif hist_sim == 'chisqr':
        for item in hist_index:
            score = hist_chisqr_sim(inhist_vec,  hist_index.get(item))
            RatingDictionary[item] = score
        topMatches = find_smallest_matches(RatingDictionary, topn)
        return topMatches

    elif hist_sim == 'inter':
        for item in hist_index:
            score = hist_intersect_sim(inhist_vec,  hist_index.get(item))
            RatingDictionary[item] = score
        topMatches = find_biggest_matches(RatingDictionary, topn)
        return topMatches

    elif hist_sim == 'bhatta':
        for item in hist_index:
            score = hist_bhatta_sim(inhist_vec,  hist_index.get(item))
            RatingDictionary[item] = score
        topMatches = find_smallest_matches(RatingDictionary, topn)
        return topMatches

def find_biggest_matches(dict, topn):
    l = []
    for key, value in sorted(dict.iteritems(), key=lambda (k, v): (v, k)):
        l.append((key, value))

    top = []
    for x in range(len(l)-topn, len(l)):
        top.append(l[x])
    return top

def find_smallest_matches(dict, topn):
    l = []
    for key, value in sorted(dict.iteritems(), key=lambda (k, v): (v, k)):
        l.append((key, value))

    top = []
    for x in range(0, topn):
        top.append(l[x])
    return top

def show_images(input_image, match_list):
    fig = plt.figure(1)
    fig.suptitle('Input')
    plt.imshow(input_image)
    plt.show()

    count = 2
    for item in match_list:
        imgText = item[0]
        img = cv2.imread(imgText)
        fig = plt.figure(count)
        fig.suptitle(imgText)
        plt.imshow(img)
        plt.show()
        count += 1

if __name__ == '__main__':
    with open(args['hist'], 'rb') as histfile:
        HIST_INDEX = pickle.load(histfile)
    sim_list = compute_hist_sim(inhist_vec, HIST_INDEX)
    for imagepath, sim in sim_list:
        print(imagepath + ' --> ' + str(sim))
    show_images(inimg, sim_list)
