# BCSP Expert case study

Project studying the differences between difficult diagnoses, particularly epithelial misplacement vs carcinoma.

### Image analysis goal

To distinguish between difficult cases and analyse the differences and difficulties, particularly cancer vs. epithelial misplacement, where human concordance may be low and locating all the suspicious regions is time-consuming and prone to subjectivity and human weakness (i.e., fatigue, overlooking areas, time constraints). We define “suspicious” here as any region of tissue within the submucosa which merits examination and most likely aids in determining the final diagnosis.


## General observations

- morphology of dysplastic epithelial cells in submucosa is more likely to be similar to that of the same cells present in the mucosa of the **same tissue sample.**
- detection of cancerous features is not sufficient; **location** is also crucial (and faster as this eliminates the need to search in mucosa). Basic spatial understanding makes the search more efficient.
- unusual areas may be along edge (interface) of the two regions. These should not be automatically considered as part of the mucosa and normal. 
- can we build a graphical model of "invasion" to study the relative locations of features? This can lead to interpretable descriptors such as "surrounded by", "next to" as well as descriptors of extent of invasion (may or may not be helpful for actual diagnoses due to problem complexity).


## Development plan

1. At low resolution, we want to use the segmentation to learn a general classifier for mucosa / submucosa.

a) Reading in labelled patches from lo resolution


2. Look at the heat map of the submucosa segmentation.

3. TBD

