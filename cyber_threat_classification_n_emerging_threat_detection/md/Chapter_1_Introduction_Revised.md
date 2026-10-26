## Chapter 1: Introduction

## 1.1 Background and Motivation

### 1.1.1 The New World Disaster of Threat Overload

In the twenty-first century, cyber threat intelligence (CTI) has become
a major concern for enterprise security. Technological progress has
connected people, systems, and organizations on an unprecedented scale.
At the same time, this connectivity has increased the number of
potential entry points available to malicious actors. Security teams
must therefore process large quantities of unstructured and rapidly
changing threat information while making decisions under time pressure.

Cyber threat intelligence can be described as the collection, analysis,
and dissemination of information about current and potential cyber
threats. It helps organizations understand weaknesses, attack vectors,
and the tactics, techniques, and procedures (TTPs) used by adversaries
to compromise systems. Unlike a simple count of exposed records after a
data breach, CTI often consists of complex narratives, technical
reports, forum posts, malware descriptions, and discussions of attacker
behaviour. Interpreting these sources requires attention to context,
terminology, and the relationships between events.

When threat information is not processed efficiently, organizations may
experience several consequences:

-   Delayed incident response because analysts must manually sift
    through large volumes of data.
-   Failure to identify zero-day exploits and newly reported
    vulnerabilities promptly.
-   More severe data breaches and longer attacker dwell time.
-   Financial losses and damage to organizational reputation.
-   Increased exposure of critical infrastructure and supply chains.
-   Greater risk from widespread ransomware campaigns.

Information overload can also reduce the productivity of security
analysts and affect the functioning of a Security Operations Center
(SOC). As communication becomes increasingly digital, threat actors
discuss their methods across public websites, hacking forums, messaging
platforms, and dark-web marketplaces. These environments may contain
vulnerability disclosures, malware source code, exploit demonstrations,
and discussions of attack campaigns. The volume and informality of this
material make automated analysis valuable, provided that the system can
distinguish meaningful threat signals from irrelevant content.

### 1.1.2 Weaknesses of Standard Threat Classification Measures

Traditional threat classification has often relied on rule-based
detection, signature-based intrusion detection systems (IDS), keyword
searches, and manual analyst review. These methods remain useful in many
security contexts, particularly when a known indicator or signature is
available. However, they can struggle when threat descriptions use
unfamiliar terminology, synonyms, obfuscated wording, or indirect
expressions.

The principal limitations are as follows:

1.  **Dependence on exact keyword matches:** a system may miss a
    relevant report when the wording differs from the terms included in
    its rules.
2.  **Limited semantic understanding:** a keyword-based method may fail
    to recognize that different phrases describe the same malicious
    behaviour.
3.  **Scalability constraints:** manual review and simple rules become
    difficult to maintain when data volumes increase.
4.  **Limited real-time monitoring:** a static process may not keep pace
    with continuously published threat information.
5.  **Weak emerging-trend detection:** conventional classification may
    identify known categories without showing that a new topic or attack
    method is becoming more common.

These limitations create a need for methods that support continuous,
automated, and scalable identification of threat indicators. A system
that depends only on known signatures can be less effective when facing
new attack vectors or evolving descriptions of familiar techniques.
Context-aware models and temporal analysis can complement existing
security controls by helping analysts prioritize information for further
investigation.

## 1.1 Background and Motivation (Continued)

### 1.1.3 Value of Computational Threat Intelligence

The growth of the internet and online communities has produced
substantial amounts of user-generated text relevant to cybersecurity.
Websites, discussion forums, messaging channels, and other online spaces
may contain conversations about:

-   Exploit code and proof-of-concept (PoC) scripts.
-   Malware updates and emerging ransomware variants.
-   Command-and-Control (C2) infrastructure.
-   Data exfiltration and negotiations with victims.
-   Vulnerabilities, attack methods, and the tools used by threat
    actors.

Computational threat intelligence applies methods from artificial
intelligence (AI), natural language processing (NLP), and data analysis
to identify useful information in these digital traces. The aim is to
transform unstructured text into actionable intelligence that can
support security analysts. Such processing may include filtering
irrelevant documents, classifying threat-related text, grouping similar
reports, and tracking changes in the frequency of topics over time.

Earlier approaches included lexicon-based analysis, Bag-of-Words
representations, TF-IDF features, and traditional machine-learning
classifiers. These techniques provide useful baselines and can be
efficient for relatively straightforward tasks. However, they often
represent text primarily through individual words or weighted terms.
This can limit their ability to model the relationships between words
and the overall meaning of a sentence, particularly in technical
cybersecurity language.

For example, two reports may describe the same behaviour using different
terminology. A model that relies on exact terms may treat them as
unrelated, even when both describe a similar technique. Conversely, a
report may contain a security-related word in a harmless context and be
incorrectly flagged. Semantic representations are intended to reduce
these problems by representing text according to contextual meaning
rather than relying only on word overlap.

### 1.1.4 The Transformer Revolution in NLP

Transformer-based architectures have significantly influenced natural
language processing. Unlike recurrent or convolutional approaches that
process information through different sequential or local operations,
transformers use attention mechanisms to model relationships between
tokens. This supports the representation of long-distance dependencies,
parallelized training, and contextual embeddings.

Models such as Sentence-BERT (SBERT) and RoBERTa can represent language
in ways that account for surrounding context. This is important for
cyber threat detection because threat descriptions are often indirect,
obfuscated, context-dependent, or not explicitly malicious when
individual words are considered in isolation.

Consider the sentence: "The actor utilized a weaponized document to
establish persistence." It does not contain some common keywords such as
"virus" or "hack," yet it describes behaviour that may be
security-relevant. A contextual model can use the relationship between
"weaponized document" and "establish persistence" to support a more
informed classification than a method that searches for a short list of
terms alone.

Transformer-based representations do not guarantee correct predictions.
Their performance depends on the data, labels, training procedure, and
evaluation method. Nevertheless, they provide a strong basis for
studying whether contextual embeddings improve the identification of
subtle cyber threat expressions.

### 1.1.5 Requirement for Emerging Threat Detection

Many classification systems produce a fixed label, such as "Malware" or
"Phishing," without explaining how the wider threat landscape is
changing. Modern cybersecurity also requires temporal awareness and
interpretable evidence. Analysts need to recognize sudden increases in
activity, understand whether a topic is new or recurring, and
distinguish a temporary fluctuation from a sustained trend.

Topic modelling can help organize large collections of documents into
groups that share semantic characteristics. In this study, BERTopic is
considered for discovering topic patterns using document embeddings,
dimensionality reduction with UMAP, and clustering with HDBSCAN. When
topic information is examined over time, it can help reveal which themes
are increasing in prominence and which may represent emerging areas of
concern.

Temporal topic analysis can provide two complementary views: the overall
importance of a topic across the dataset and the local changes that may
indicate a developing trend. These results should be interpreted
alongside the underlying documents and the limitations of the available
data. A rise in topic frequency is a signal for analyst attention, not
proof by itself that an attack is occurring or that a threat is new.

## 1.1 Background and Motivation (Continued)

### 1.1.6 Transitioning from Detection to Mitigation

A classification result is most useful when it can support a practical
decision. If an analytical system stops after assigning a label, the
analyst may still need to search for supporting evidence, assess
severity, and determine the next action. The proposed study therefore
connects detection outputs with a web-based analyst dashboard intended
to make results easier to review and act upon.

The dashboard is planned to map predicted severity and classification
results to suitable intervention categories. Its intended functions
include:

-   Visual alerts for potentially critical or emerging threats.
-   Recommendations that help analysts review known techniques and
    possible responses.
-   Interactive guidance mapped to the MITRE ATT&CK knowledge base.
-   Downloadable reports to support documentation and further
    investigation.
-   Trend visualizations that help users inspect topic activity over
    time.

These features are designed as decision-support mechanisms rather than
automatic mitigation. Human analysts remain responsible for validating
model outputs, considering organizational context, and deciding whether
a response is appropriate. This approach moves the system from passive
analytics toward practical support while preserving human oversight.

<div align="center">
  <img
    src="./assets/Figure 1.1 Proposed Workflow for Cyber Threat Classification and Emerging Threat Detection.png"
    alt="Proposed Workflow for Cyber Threat Classification"
    style="width: 75%; height: auto;"
  />
</div>

## 1.2 Problem Statement

Despite progress in computational approaches to threat detection,
current cyber threat intelligence systems can have several important
limitations.

**1. Binary simplification.** Some systems reduce threat relevance to a
binary decision and do not represent differences in severity or the
progression of a threat. This can make it difficult to prioritize
reports that vary in urgency or operational impact.

**2. Shallow context modelling.** Traditional machine-learning models
often use features such as TF-IDF and may not fully represent contextual
relationships in cybersecurity language. They can miss semantically
similar descriptions or overemphasize terms that appear in benign
contexts.

**3. Limited attention to temporal indicators.** Attack frequency,
changes over time, and the evolution of topics may be overlooked. As a
result, a system may classify individual documents without identifying a
sudden increase in discussion about a particular malware campaign or
technique.

**4. Lack of explainability.** Complex models may not provide analysts
with a clear explanation for a prediction. Without accessible supporting
evidence, users may find it difficult to assess why a sentence was
flagged or how much confidence to place in the result.

**5. Lack of intervention mapping.** Detection results are not always
translated into practical support such as visual alerts, trend charts,
structured guidance, or downloadable reports. The analyst may have to
move between separate tools to interpret the result and determine an
appropriate next step.

Taken together, these limitations motivate the development and
evaluation of a cyber threat analysis framework that combines contextual
classification, technique mapping, temporal topic analysis, and a
user-oriented dashboard. The system is intended to help analysts
identify relevant information earlier and review potential emerging
threats in a structured way.

## 1.3 Research Objectives

The overall aim of this study is to develop and evaluate an automated
AI-based system that combines transformer-based language representations
with topic modelling to classify cyber threat information and identify
possible emerging campaigns. The proposed framework emphasizes
contextual understanding, temporal analysis, and the presentation of
results in a form that can assist human analysts.

The specific objectives are:

1.  To create a binary relevance-classification model that distinguishes
    relevant from irrelevant cybersecurity text using Sentence-BERT
    representations.
2.  To develop and evaluate a multi-class Support Vector Machine (SVM)
    classifier that maps relevant text to MITRE ATT&CK techniques.
3.  To apply BERTopic, using UMAP and HDBSCAN, to discover and group
    semantically related topics in the collected text.
4.  To design a temporal, rule-based component that identifies
    potentially emerging threat trends by examining changes in topic
    activity.
5.  To develop a web-based recommendation and analyst-support system
    using FastAPI.
6.  To evaluate model performance under class-imbalanced conditions
    using accuracy and macro-averaged F1-score.

These objectives are connected within a single workflow. The relevance
classifier is intended to filter text for further analysis. The
multi-class classifier provides more specific technique-level
information, while topic modelling groups related documents and temporal
analysis highlights changes in those groups. The dashboard then presents
selected outputs to support analyst review.

The evaluation will need to consider not only overall predictive
performance but also performance across classes. Accuracy alone can be
misleading when some labels occur much more frequently than others.
Macro-F1 gives each class equal weight in the calculation of the average
F1-score and can therefore provide additional insight into performance
on less frequent classes.

## 1.4 Research Questions

The study is guided by the following research questions:

1.  To what extent can transformer-based contextual embeddings identify
    subtle expressions of cyber threats?
2.  Can multi-class classification of MITRE ATT&CK techniques improve
    the specificity and actionability of the information presented to
    analysts?
3.  Can temporal analysis of BERTopic outputs produce interpretable
    indications of potentially emerging threats?
4.  How can classification and trend-analysis results be mapped to
    structured mitigation guidance in a web-based dashboard?
5.  How well does the proposed model perform when the dataset contains
    an imbalance between classes?

The first question examines contextual understanding, while the second
focuses on the value of more detailed technique labels. The third
addresses whether topic changes over time can provide useful signals for
analysts. The fourth considers how model outputs can be communicated
through an operational interface. The fifth evaluates the reliability of
the framework when some classes are represented by fewer examples than
others.

## 1.5 Scope and Significance of the Study

### 1.5.1 Scope of the Study

The study focuses on English-language cybersecurity text obtained from
PDF reports. Its planned scope includes binary relevance classification,
multi-class threat-technique classification, semantic text
representation, BERTopic-based topic discovery, temporal trend analysis,
and an interactive web dashboard.

The main boundaries of the study are as follows:

-   **Data type:** English-language cybersecurity text extracted from
    PDF reports.
-   **Classification:** binary relevant-versus-irrelevant prediction and
    multi-class technique classification.
-   **Semantic processing:** contextual embeddings used to represent the
    meaning of text.
-   **Topic analysis:** BERTopic used to group semantically related
    documents.
-   **Temporal analysis:** topic activity examined over time to identify
    possible emerging trends.
-   **User interface:** an interactive dashboard intended to present
    results and analyst-support recommendations.

The proposed system is intended to function as a supportive AI tool
rather than an autonomous diagnostic or mitigation system. Its
predictions should be checked against source material and interpreted in
the context of the organization using it. The scope described here does
not imply that every threat can be detected or that every emerging trend
can be confirmed automatically.

### 1.5.2 Scientific Significance

The scientific value of the study lies in investigating how contextual
text representations, technique-level classification, and temporal topic
analysis can be combined in a common cyber threat intelligence workflow.
The framework brings together document-level prediction and broader
topic-level analysis, allowing the study to examine both individual
threat expressions and changes in the themes represented in the dataset.

### 1.5.3 Technological Significance

From a technological perspective, the proposed framework combines
transformer embeddings, a multi-class classifier, topic modelling,
temporal rules, and a FastAPI-based web interface. This design is
intended to connect analytical components with a practical means of
reviewing their outputs. The effectiveness of this integration must be
assessed through implementation and empirical evaluation.

### 1.5.4 Ethical and Societal Significance

The study supports the responsible use of AI in cyber defence by
positioning the system as assistance for human analysts rather than a
replacement for professional judgment. This distinction is important
because incorrect classifications or misleading trend signals could lead
to wasted effort or inappropriate decisions. Clear presentation of
evidence, acknowledgement of uncertainty, and human review are therefore
important considerations.

At a societal level, earlier identification and structured review of
relevant threat information may help organizations investigate risks
before they cause greater harm. The proposed dashboard is intended to
support this process by presenting classification outputs,
emerging-topic indicators, and guidance in a unified interface.

## 1.6 Research Contributions

The expected contributions of this thesis are summarized below.

**1. Transformer-based relevance classification.** The study proposes a
fine-grained approach to identifying cybersecurity text that is relevant
to threat analysis. Sentence-BERT representations are used as the basis
for distinguishing relevant material from irrelevant content.

**2. Technique-level classification.** The framework includes
multi-class classification intended to map text to MITRE ATT&CK
techniques. This adds a more specific layer of information beyond a
general relevance label and can help analysts organize findings by
technique.

**3. Topic discovery and explainability support.** BERTopic is
incorporated to group semantically related text. Topic representations
and their associated documents can help analysts inspect the content
underlying a discovered theme rather than relying only on a predicted
class.

**4. Temporal emerging-threat analysis.** A rule-based component is
proposed to examine changes in topic activity over time. Its purpose is
to flag possible increases or emerging patterns for further
investigation, not to treat frequency changes as conclusive proof of an
attack.

**5. Analyst-support dashboard.** The work proposes a web-based
interface that presents results through visual alerts, trend
information, MITRE ATT&CK guidance, and downloadable reports. The
interface is intended to connect analytical outputs with a more
structured review process.

**6. Experimental evaluation.** The framework is to be evaluated using
accuracy and macro-F1, with attention to class imbalance. This
evaluation is important for understanding whether performance remains
useful when the distribution of labels is uneven.

**7. Integrated and scalable architecture.** The proposed design
connects text processing, classification, topic modelling, temporal
analysis, and a FastAPI-based interface. The architecture is intended to
provide a foundation for further testing and possible adaptation to
security operations.

## 1.7 Chapter Summary

This chapter has introduced the motivation for automated cyber threat
intelligence analysis and described the challenges created by high
volumes of unstructured threat information. It has discussed limitations
of keyword-based and traditional classification approaches, including
their restricted contextual understanding, scalability challenges, and
limited ability to identify changes in threat topics over time.

The chapter has presented the proposed use of transformer-based
contextual embeddings for relevance classification, a multi-class SVM
for technique mapping, and BERTopic with UMAP and HDBSCAN for semantic
topic discovery. It has also described the role of temporal rules in
identifying possible emerging trends and the purpose of a web-based
dashboard in presenting results for human review.

Finally, the chapter has stated the problem, research objectives,
research questions, scope, significance, and expected contributions of
the study. The next stages of the thesis should detail the related work,
methodology, implementation, and experimental evaluation needed to
assess the proposed framework.
