I run the auto inversion procedure and annotate the differences from my taste (so effectively I'm giving ground truth to which the auto detection should ideally converge). This ground truth is constant.
The algorithm is not and will evolve as we're trying to reach that ground truth.

The current algorithm was tuned in steps, at first I gave you annotations for one film roll, you tuned something telling results only in chat session, it improved detection a bit. Then I annotated one more roll, you calibrated something again etc.

Now I realize that we've lost track over this. You did something and I don't know exactly what and what convergence metrics were obtained, i.e. how closely you were able to get to my annotations. But I should know the metric so we can see we're improving as we change the algorithm.

Now we've reached the point that you changed the algo and I must reannotate all rolls afresh (the current annotations are all wrong and I'll delete them...). Before I spend time reannotating I want to make sure that we'll do the right thing and my complex labor for annotating won't go in vain.

Therefore, my requirements:
- Annotations serve as ground truth and the format stays the same, in folder `tests/fixtures/rolls`
- Calibrations though need to be totally reworked:
    - Each calibration session must be a separate folder in this repo, not some fleeting chat output in your window or memory. Let it be folders 'tests/calibrations/<session-date-time-number>/`
    - Each session must clearly indicate which source rolls (i.e. annotations) it used for calibration. I.e. one session could have been tuning global params to just one roll's annotations. Was stored as one folder. Then I add another roll with annotations and then rerun the calibration session, but this time not JUST FOR THE SECOND ROLL, BUT FOR BOTH ROLLS, OLD AND NEW, with the goal of finding algo params that fit best BOTH rolls together.
    - Each calibration must record INPUT CRITERIA, i.e. which tightness of params fitting was used, which algo of fitting, which epsilon of convergence etc.
    - As I care about the final look only, the designed convergence metrics should be designed to be mirroring closeness of fitting to my params, i.e. be totally independent of actual inversion algorithm, because whatever it may be, in the end we're comparing one version of an image with another. So let's spend time designing these metrics. (well maybe they can be the current ones, I just don't fully know them, you did something silently...)
    - Each calibration must record OUTPUT RESULTS, both total accross all rolls and then total for each roll and then per each frame of the roll. I.e. metrics must show how closely this calibration session was able to tune the algo to reproduce my manual annotations of the rolls in question and also it should be easy to find bad fitting frames to understand where to look for improvement
    - So as time goes and we improve the algo or I add more rolls with annotations and rerun the sessions, I should be able to meaningfully compare history of fitting results.
    - should support tuning cropping too, also with metrics how close it was able to get to my annotations. In fact I first will annotate just crops because correct cropping heavily affects further algo's work.