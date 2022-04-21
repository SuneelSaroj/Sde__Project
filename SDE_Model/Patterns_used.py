

import spacy
import re
import itertools
import pickle
import nltk
from Preprocessing import preprocessing
import Preprocessing
import importlib
import requests
from langdetect import detect
import json


from enum import Enum


class EXTRACTION_MODE(Enum):
    APP_DESCRIPTION = 1
    USER_REVIEWS = 2


class Patterns_used:
    def __init__(self, appName, sent_id, clean_sents, unclean_data):
        self.appName = appName
        self.sentid = sent_id
        self.clean_sentences = clean_sents
        self.unclean_sentences = unclean_data

    def ExtractFeatures_Analyzing_Sent_POSPatterns(self):
        raw_data_sent_patterns, remaining_sents, sent_with_features = self.Extract_AppFeatures_with_Patterns()
        raw_data_pos_patterns, sents_features = self.Extract_AppFeatures_with_POSPatterns(
            sent_with_features)
        extracted_app_features = raw_data_sent_patterns + raw_data_pos_patterns

        return sents_features

    def SaveExtractedFeatures(self, extracted_features):
        file_path = self.appId.upper() + "_EXTRACTED_APP_FEATURES_"
        if self.extraction_mode.value == EXTRACTION_MODE.APP_DESCRIPTION.value:
            file_path = file_path + "DESC.pkl"
        elif self.extraction_mode.value == EXTRACTION_MODE.USER_REVIEWS.value:
            file_path = file_path + "REVIEWS.pkl"

        with open(file_path, 'wb') as fp:
            pickle.dump(extracted_features, fp)

    def Extract_Features_with_single_POSPattern(self, pattern_1, tag_text):
        match_list = re.finditer(pattern_1, tag_text)

        app_features = []

        for match in match_list:
            app_feature = tag_text[match.start():match.end()]
            feature_words = [w.split("/")[0] for w in app_feature.split()]
            app_features.append(' '.join(feature_words))

        return(app_features)

    def Extract_AppFeatures_with_POSPatterns(self, sent_with_features):
        app_features_pos_patterns = []

        pos_patterns = [r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)",
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",


                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",


                        r"[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/PRON\s+[a-zA-Z-]+\/(NOUN)",


                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/(NOUN)\s+(with|to)\/(ADP|PRT)\s+[a-zA-Z-]+\/(NOUN)",



                        r"[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/(DET)\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/ADP\s+[a-zA-Z-]+\/(NOUN)",

                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/CONJ\s+[a-zA-Z-]+\/ADJ",

                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(PRON)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",


                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(ADP)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",


                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/DET\s+[a-zA-Z-]+\/(ADJ|NOUN)\s+[a-zA-Z-]+\/(NOUN)"
                        ]

        for sentence_id in sent_with_features.keys():
            sent_info = sent_with_features[sentence_id]
            sent = sent_info['clean_sent']

            if len(sent_info['extracted_features']) == 0:
                extracted_features = []
                sent_tokens = nltk.word_tokenize(sent)
                tag_tokens = nltk.pos_tag(sent_tokens, tagset='universal')
                tag_text = ' '.join(
                    [word.lower() + "/" + tag for word, tag in tag_tokens])

                rule_counter = 1
                for pattern in pos_patterns:

                    rule_name = 'POS_R%d' % (rule_counter)
                    raw_data = self.Extract_Features_with_single_POSPattern(
                        pattern, tag_text)
                    if len(raw_data) != 0:
                        app_features_pos_patterns.extend(raw_data)
                        for extracted_feature in set(raw_data):
                            extracted_features.append(
                                (extracted_feature, rule_name))

                    rule_counter = rule_counter + 1

                if len(extracted_features) > 0:
                    sent_feature_info = sent_with_features[sentence_id]
                    sent_feature_info['extracted_features'] = list(
                        set(extracted_features))
                    sent_with_features[sentence_id] = sent_feature_info

        return(app_features_pos_patterns, sent_with_features)

    def Pattern_Case1(self, tag_text):
        raw_data = []
        regex_case1 = r"[a-zA-Z-]+\/(NOUN)(\s+,\/.)?\s+(and)\/CONJ\s+[a-zA-Z-]+\/(ADJ)(\s+[a-zA-Z-]+\/(NOUN))"
        match = re.search(regex_case1, tag_text)
        if match != None:
            matched_text = tag_text[match.start():match.end()]
            words = [w.split("/")[0] for w in matched_text.split()
                     if w.split("/")[1] not in ['.', 'CONJ']]
            raw_data.append(words[0] + " " + ' '.join(words[2:]))
            raw_data.append(words[1] + " " + ' '.join(words[2:]))

        return(raw_data)

    def Pattern_Case2(self, tag_text):
        raw_data = []

        regex_case2 = r"[a-zA-Z-]+\/(ADJ|NOUN)(\s+[a-zA-Z-]+\/(NOUN)\s+,\/.)+(\s+[a-zA-Z-]+\/(NOUN))?\s+and\/CONJ\s+[a-zA-Z-]+\/(NOUN)"
        match = re.search(regex_case2, tag_text)
        if match != None:
            matched_text = tag_text[match.start():match.end()]
            words = matched_text.split()

            first_word = words[0].split("/")[0]
            last_word = words[len(words)-1].split("/")[0]

            enumeration_words = [w.split('/')[0] for index, w in enumerate(matched_text.split(
            )) if index not in [0, len(words)-1] and w.split("/")[1] not in ['.', 'CONJ', 'PRON']]
            raw_data.append(first_word + " " + last_word)

            raw_data += [first_word + " " + w2 for w2 in enumeration_words]

        return(raw_data)

    def Pattern_Case3(self, tag_text):
        raw_data = []
        regex_case3 = r"[a-zA-Z-]+\/(VERB|NOUN)\s+and\/CONJ\s+[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/(NOUN|VERB)\s+and\/CONJ\s+[a-zA-Z-]+\/(NOUN|VERB)"
        match = re.search(regex_case3, tag_text)
        if match != None:
            matched_text = tag_text[match.start():match.end()]
            words = matched_text.split()
            words = [w.split("/")[0] for w in words]
            l1 = [words[0], words[2]]
            l2 = [words[3], words[5]]
            list_raw_data = list(itertools.product(l1, l2))
            raw_data = [feature_words[0] + " " + feature_words[1]
                        for feature_words in list_raw_data]

        return(raw_data)

    def Pattern_Case4(self, tag_text):
        raw_data = []
        regex_case4 = r"[a-zA-Z-]+\/(VERB|NOUN|ADJ)\s+and\/CONJ\s+[a-zA-Z-]+\/(VERB|NOUN|ADJ)\s+[a-zA-Z-]+\/ADP((\s+[a-zA-Z-]+\/(NOUN|VERB))(\s+[a-zA-Z-]+\/(ADP)))?\s+[a-zA-Z-]+\/(NOUN|VERB)"
        regex_case4 += "(\s+,\/.)?\s+(including\/[a-zA-Z-]+)((\s+[a-zA-Z-]+\/(VERB|NOUN))+\s+,\/.)+\s+[a-zA-Z-]+\/(NOUN|VERB)\s+(and\/CONJ)\s+[a-zA-Z-]+\/(NOUN|VERB)"

        match = re.search(regex_case4, tag_text)

        if match != None:
            matched_text = tag_text[match.start():match.end()]
            words = matched_text.split()
            words = [w.split("/")[0] for w in words]

            feature_word1 = words[0]
            feature_word2 = words[2]

            feature_list1 = [words[0], words[2]]

            count = 0
            feature_list2 = []
            fwords = []
            for i in range(3, len(words)):
                if i < len(words)-1:
                    if words[i+1] == "," and count == 0:
                        feature_list2.append(words[i])
                        count = count + 1
                    elif count == 1:
                        if words[i] != "including" and words[i] != ',':
                            fwords.append(words[i])
                        if words[i] == ",":
                            if len(fwords) > 0:
                                feature_list2.append(' '.join(fwords))
                            fwords = []

            feature_list2.append(words[len(words)-1])
            feature_list2.append(words[len(words)-3])

            list_raw_data = list(
                itertools.product(feature_list1, feature_list2))

            raw_data = [feature_words[0] + " " + feature_words[1]
                        for feature_words in list_raw_data]

        return(raw_data)

    def Pattern_Case5(self, tag_text):
        raw_data = []
        regex_case5 = r"[a-zA-Z-]+\/(VERB|NOUN|ADP)\s+,\/.\s+[a-zA-Z-]+\/(VERB|NOUN)\s+and\/CONJ\s+[a-zA-Z]+\/(VERB|NOUN|ADJ)\s+[a-zA-Z-]+\/(NOUN|VERB|ADJ)\s+(as\/ADP)\s+"
        regex_case5 += "[a-zA-Z-]+\/(ADJ|NOUN|VERB)(\s+[a-zA-Z-]+\/(NOUN|VERB)\s+,\/.)+\s+[a-zA-Z-]+\/(NOUN|VERB)\s+(and\/CONJ)"
        regex_case5 += "\s+[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/(NOUN|VERB)"
        match = re.search(regex_case5, tag_text)
        if match != None:
            match_text = tag_text[match.start():match.end()]
            words_with_tags = match_text.split()
            words = [w.split("/")[0] for w in words_with_tags]

            feature_list1 = [words[0], words[2]]
            feature_list2 = [words[4] + " " +
                             words[5], words[7] + " " + words[8]]
            feature_list3 = [words[10], words[12], words[14] + " " + words[15]]
            list_raw_data = list(
                itertools.product(feature_list1, feature_list3))
            raw_data = [feature_words[0] + " " + feature_words[1]
                        for feature_words in list_raw_data]
            raw_data = raw_data + feature_list2

        return(raw_data)

    def Extract_AppFeatures_with_Patterns(self):
        raw_app_features_sent_patterns = []
        clean_sents_wo_sent_patterns = []
        sents_with_extracted_features = {}

        sent_id = 0

        for sent_index in range(0, len(self.clean_sentences)):
            sent = self.clean_sentences[sent_index]
            sent_features = []
            try:
                sent_text = self.unclean_sentences[sent_index]
            except KeyError as ex:
                print(sent_id)

            sent_tokens = nltk.word_tokenize(sent)
            tag_tokens = nltk.pos_tag(sent_tokens, tagset='universal')
            tag_text = ' '.join(
                [word.lower() + "/" + tag for word, tag in tag_tokens])

            sent_pattern_found = False

            raw_data_case1 = self.Pattern_Case1(tag_text)

            if len(raw_data_case1) != 0:
                raw_app_features_sent_patterns.extend(raw_data_case1)

                for i in range(0, len(raw_data_case1)):
                    extracted_feature = raw_data_case1[i]
                    rule = 'SS_R1'
                    sent_features.append((extracted_feature, rule))

                sent_pattern_found = True

            raw_data_case2 = self.Pattern_Case2(tag_text)

            if len(raw_data_case2) != 0:
                raw_app_features_sent_patterns.extend(raw_data_case2)

                for i in range(0, len(raw_data_case2)):
                    extracted_feature = raw_data_case2[i]
                    rule = 'SS_R2'
                    sent_features.append((extracted_feature, rule))

                sent_pattern_found = True

            raw_data_case3 = self.Pattern_Case3(tag_text)
            if len(raw_data_case3) != 0:
                raw_app_features_sent_patterns.extend(raw_data_case3)

                for i in range(0, len(raw_data_case3)):
                    extracted_feature = raw_data_case3[i]
                    rule = 'SS_R3'
                    sent_features.append((extracted_feature, rule))

                sent_pattern_found = True

            raw_data_case4 = self.Pattern_Case4(tag_text)
            if len(raw_data_case4) != 0:
                raw_app_features_sent_patterns.extend(raw_data_case4)
                sent_features.extend(raw_data_case4)

                for i in range(0, len(raw_data_case4)):
                    extracted_feature = raw_data_case4[i]
                    rule = 'SS_R4'
                    sent_features.append((extracted_feature, rule))

                sent_pattern_found = True

            raw_data_case5 = self.Pattern_Case5(tag_text)
            if len(raw_data_case5) != 0:
                raw_app_features_sent_patterns.extend(raw_data_case5)

                for i in range(0, len(raw_data_case5)):
                    extracted_feature = raw_data_case5[i]
                    rule = 'SS_R5'
                    sent_features.append((extracted_feature, rule))

                sent_pattern_found = True

            if sent_pattern_found == False:
                clean_sents_wo_sent_patterns.append(sent)

            sents_with_extracted_features[sent_index] = {
                'sentence_text': sent_text, 'clean_sent': sent, 'extracted_features': sent_features}

        return(raw_app_features_sent_patterns, clean_sents_wo_sent_patterns, sents_with_extracted_features)
