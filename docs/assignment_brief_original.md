|   |
| - |

**Weighting %:**

|   |
| - |

30

|   |
| - |

**Submission deadline (for students):  **

|   |
| - |

20/08/2026

|   |
| - |

**Authorship:**

** **

|   |
| - |

Individual

 

|   |
| - |

**Target date for returning marked coursework: **

|   |
| - |

29/08/2026

|   |
| - |

**Tutor setting the work:**

|   |
| - |

Nickolay Korabel & William Alston

|   |
| - |

**Number of hours you are expected to work on this assignment:**

|   |
| - |

40

|   |
| - |

** **

---

**This Assignment assesses the following module Learning Outcomes (from Definitive Module Document):**

** **

1.     Demonstrate knowledge and understanding of real−world applications of data science

analysis techniques and Al.

2.     Demonstrate knowledge and understanding in downstream techniques and methods to generate novel insights and results from real−world data

3.     Be able to demonstrate the ability to use a range of current data analysis techniques and methods using Al.

4.     Be able to provide in−depth downstream analysis of a range of data types to extract novel information and make useful predictions and data−driven decisions.

---

|   |
| - |

**In this assignment, you will build and evaluate a compact biomedical image-analysis pipeline that combines the methods covered in this part of the module: a local multimodal large language model (VLM), classical image processing, and a U-Net segmentation network.**

**You will be assigned one imaging modality and a small dataset, and you will move images through the pipeline: raw image → segmentation → quantitative region features → structured JSON record → short narrative — producing outputs that are auditable rather than free-text guesses.**

 

This assignment is based on topics covered in Lecture/Lab 2 (multimodal LLMs for medical image description), Lecture/Lab 3 (classical image features plus LLM interpretation), Lecture/Lab 4 (U-Net for biomedical segmentation), and Lecture/Lab 5 (hybrid pipelines and medical AI in practice).

**Objective:** Build a local, hybrid biomedical image-analysis system for an assigned modality; compare a direct visual language model (VLM) description against a numbers-first description (classical methods); train and evaluate a small U-Net; and assemble the whole pipeline into an auditable per-image record.

All large language models are run locally via Ollama, as in the labs. The outputs are for educational use only — none of the models are cleared for clinical use, and hallucinations in a medical context can cause harm. Starter scripts (a U-Net skeleton and a README template) are provided in the Assignment Supplementary Information section on Canvas.

 

**Tasks:**

**Task 1: Data preparation and multimodal LLM description.** Download the dataset https\://github.com/Nickolay-K/Assingnment-3-dataset, convert the images to grayscale, resize them to a common size (256×256), and produce a short EDA (a sample of images and an intensity histogram). Then send a representative image to a local multimodal model (llama3.2-vision) via Ollama. Engineer a structured prompt that anchors the model as descriptive rather than diagnostic, forces a JSON record (modality, tissue\_type, notable\_features, image\_quality), and explicitly permits “uncertain”; compare it against a naive prompt, and briefly show that repeated runs are not identical. Record the optimised prompt(s). They must appear in the report.

**Task 2: Classical features and LLM interpretation.** Using scikit-image, apply Otsu thresholding and morphological cleanup, label the connected components, and compute a per-object feature table with regionprops\_table (area, eccentricity, solidity, mean intensity, and so on). Convert the table into a short natural-language summary and pass that summary (numbers only; the model never sees the image) to a local LLM, requesting a one-paragraph description and a JSON record (n\_objects, density\_class, shape\_regularity, quality\_flag). Compare this numbers-first description against the direct image description from Task 1.

**Task 3: U-Net segmentation.** Train the provided small U-Net (PyTorch) on the mini-dataset for a modest number of epochs. Evaluate it with mean Dice and IoU on the held-out validation split, and show the input, ground-truth mask, and prediction side by side for at least three validation images.

**Task 4: Hybrid pipeline.** Run the full pipeline on the unseen test images: U-Net mask → regionprops feature table → LLM structured JSON record (image\_id, n\_objects, mean\_area, density\_class, quality\_flag) → one-paragraph narrative. Aggregate the JSON records across all test images into a pandas DataFrame and save it as a CSV.

**Task 5: Write a report (max 4 pages)** describing the results of Tasks 1-4. Include sample images and histograms; U-Net input, ground-truth, and prediction panels; loss and Dice curves; a table of evaluation metrics across models and losses; example JSON records and narratives. A critical analysis of your work and results is essential to scoring high marks: do not merely state numerical results, but discuss why the results are as they are, based on your code and models, and what your design choices traded off. References should be no more than 0.5 pages; cite papers where appropriate.

**Extension for extra credit:**

•  Robustness: corrupt one image (heavy blur, low contrast, or added noise) and trace how the corruption propagates through the mask, the feature table, and the narrative, identifying the earliest stage at which it becomes detectable.

•  Model or loss comparison: compare at least two vision models on the description step, or run a loss ablation on the U-Net (BCE vs Dice vs BCE+Dice) and report which gives the best validation Dice.

•  Foundation model: replace the trained U-Net with a pretrained medical segmentation model (for example MedSAM) and compare its masks against your own.

 

**Answer these questions in your report** (ordered from more straightforward to more open-ended):

**1. **Which description is more useful, and which is more trustworthy, the direct VLM description (Task 1) or the numbers-first description (Task 2) and why?

**2. **Did the U-Net improve on classical Otsu segmentation for your modality? Give one example image where each approach did better.

**3. **Report your U-Net’s Dice and IoU. What do these numbers mean, and where does the model tend to make its mistakes (which images or regions)?

**4. **Where in the pipeline can the LLM hallucinate, and what design choices reduce that risk? Why does keeping the structured JSON as the “source of truth” help?

**5. **Considering accuracy, auditability, and the limits of your dataset, would you trust any part of this system in a real clinical setting? What single change would most improve trustworthiness?

 

---

|   |
| - |

**Submission Requirements:**

**Report: **a PDF document only, maximum 4 pages. Describe the analysis steps you carried out, why you took the approach you did, and how you interpreted the results, and answer the questions above. You must also include the optimised prompts used at each LLM step, together with the required structured outputs and visualisations.

**Code: **you can submit the code as an additional file or share a link to a Colab or GitHub repository with a clear README. If we cannot access the code you may lose marks in the code section. The code must include all the figures, models, and numerical values presented in the report, and it must run when we test it.

|   |
| - |

---

 

**Marks awarded for:**

**1. Code submission: (60%)**

**Completion of the tasks (50%)**

·       **Data preparation and EDA — 8**

·       **Multimodal LLM outputs and optimised prompts — 10**

·       **Classical feature extraction and interpretation — 10**

·       **U-Net training and Dice/IoU evaluation — 12**

·       **Hybrid pipeline producing a structured JSON record, a narrative, and an aggregated CSV — 10**

**Code quality and annotation (10%)**

·       **Developing functions to structure the pipeline — 5**

·       **Annotating your code so that another user can understand and re-run it — 5**

**2. Report (max 4 pages): (40%)**

**Discussion of the analysis and inferences (30%)**

·       **A brief overview of the methods — not textbook definitions — 6**

·       **Quality of the structured outputs and the prompts used — 8**

·       **Comparison of the methods and answers to the questions — 16**

**Quality of the report writing (10%)**

·       **How well the document flows and how easy it is to follow — 4**

·       **The document is correctly formatted and figures are used appropriately — 3**

·       **The appropriate references are used throughout — 3**

 

**Weights are marks out of 100 for the assignment as a whole (the assignment itself is worth 30% of the module). Extension work adds up to 5 bonus marks, capped at 100, and only where the five completion elements are complete.**

 

|   |
| - |

** **

---

**Type of Feedback to be given for this assignment:**

** **

A comment and grade on each of the above sections. Written feedback on report and code, following rubric breakdown.



---

|   |
| - |

**Additional information:**

·       Regulations governing assessment offences including Plagiarism and Collusion are available from [https://www.herts.ac.uk/\_\_data/assets/pdf\_file/0007/237625/AS14-Apx3-Academic-Misconduct.pdf](https://www.herts.ac.uk/__data/assets/pdf_file/0007/237625/AS14-Apx3-Academic-Misconduct.pdf)(UPR AS14).

·       Guidance on avoiding plagiarism can be found here: [https://herts.instructure.com/courses/61421](https://herts.instructure.com/courses/61421) (see the **Referencing **section)

·       For **postgraduate modules**:

o   a score of 50% or above represents a pass mark.

o   late submission of any item of coursework for each day or part thereof (or for hard copy submission only, working day or part thereof) for up to five days after the published deadline, coursework relating to modules at Level 7 submitted late (including deferred coursework, but with the exception of referred coursework), will have the numeric grade reduced by 10 grade points until or unless the numeric grade reaches or is 50. Where the numeric grade awarded for the assessment is less than 50, no lateness penalty will be applied. 



**AI imaging Rubric**

| **Criteria**                         | **Ratings**             | **Points**    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                     |          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                     |          |                                                                                                                                                                                                                                                                |                  |         |
| ------------------------------------ | ----------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------- |
| **Completion of the pipeline tasks** | view longer description | **Excellent** | You have attempted and completed all five tasks and they run end-to-end. Data is preprocessed (grayscale, 256×256) with a proper EDA; your llama3.2-vision prompt is engineered to be descriptive-not-diagnostic and returns valid JSON that permits "uncertain", with a naive comparison and run-to-run variability shown; classical Otsu + regionprops feed a numbers-only LLM summary and JSON; the U-Net trains and is evaluated with mean Dice and IoU plus side-by-side panels; and the full hybrid pipeline produces per-image JSON, narratives, and an aggregated CSV on the unseen test set. | **50 to >37.5 pts** | **Good** | You have attempted all of the tasks but some are incomplete or suboptimal — e.g. prompts return JSON inconsistently, the classical or hybrid step has gaps, or the U-Net trains but its evaluation/panels are thin. The pipeline mostly works but is not fully assembled or optimised.25 to >0 pts — You have only partially completed the tasks. Major components are missing or do not run (e.g. no working U-Net, no LLM JSON, or no end-to-end pipeline and CSV). | **37.5 to >25 pts** | **Poor** | You have only partially completed the tasks.                                                                                                                                                                                                                   | **25 to >0 pts** | /50 pts |
| **Code quality and annotation**      | view longer description | **Excellent** | You have structured the pipeline into your own well-chosen functions with minimal duplication, and annotated the code (comments / README) so another user can install the dependencies (Ollama, models), understand it, and re-run it as submitted                                                                                                                                                                                                                                                                                                                                                    | **10 to >7.5 pts**  | **Good** | Your code works but is partly ad hoc — some functionalisation with duplication or long monolithic cells — and annotation leaves gaps a reader must fill; minor setup friction.                                                                                                                                                                                                                                                                                        | **7.5 to >5 pts**   | **Poor** | Your code is difficult to follow, you have not used functions where appropriate, and it is poorly annotated or cannot be re-run without significant effort.                                                                                                    | **5 to >0 pts**  | /10 pts |
| **Discussion of the analysis**       | view longer description | **Excellent** | You briefly explain the methods as used here (not textbook definitions), present your optimised prompts and valid structured outputs, and critically discuss the results: why they came out as they did given your code and model choices, what your design choices traded off, and thoughtful, evidence-based answers to all five questions (VLM vs numbers-first usefulness and trust; U-Net vs Otsu with examples; Dice/IoU meaning and failure modes; where the LLM can hallucinate and how JSON-as-source-of-truth mitigates it; clinical trust and the single highest-impact change).           | **30 to >20 pts**   | **Good** | You answer most of the questions and go somewhat beyond restating numbers, but the analysis is shallow in places, not all five questions are fully addressed, and prompts/outputs or trade-off discussion are partly missing.                                                                                                                                                                                                                                         | **20 to >10 pts**   | **Poor** | There is little or no discussion of the results, the questions are largely unanswered, prompts/structured outputs are missing, and you have not identified why the results occur or how to improve trustworthiness.                                            | **10 to >0 pts** | /30 pts |
| **Quality of the report writing**    | view longer description | **Excellent** | The report is nicely set out, easy to follow. Appropriate use of figures and references. Excellent use of English and the correct technical language is used.                                                                                                                                                                                                                                                                                                                                                                                                                                         | **10 to >7.5 pts**  | **Good** | The report is readable, good English is used and appropriate references have been used. You have shown some figures to illustrate the arguments of the report.                                                                                                                                                                                                                                                                                                        | **7.5 to >5 pts**   | **Poor** | The report is not easy to follow the structure of. You have not used any figures and you have missed important references in this research field.  The use of AI text generation tools to write the report will result in zero marks for the whole assignment. | **5 to >0 pts**  | /10 pts |