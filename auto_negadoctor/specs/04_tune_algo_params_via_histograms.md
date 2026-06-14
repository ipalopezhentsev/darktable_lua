Currently the negative inversion feature works and was tuned by you via my manual annotations here: auto_negadoctor\tests\fixtures\annotations\2026-06-13_wb_print_roll
But it still cannot reach my manual corrections.
You tried many times but hit some wall.

And I think I start understanding why.
The output params that we seek via algorithm and to which I make my manual annotations correcting the errors are essentially two groups:
1) exposure
2) white balance for shadows/highlights. 

WB is not just about hue, it also encodes 'power' of this color. changing wb changes exposure and necessitates nudging exposure afterwards. Or vice versa I can make exposure darker but then compensate it with choosing lighter colors for WB patches.

And here comes the wall you hit I think. So you have my manual params and try to reach them via your algorithm. But due to the aforementioned duality, there can be inconsistencies to my edits, I mean the picture itself that I obtain via my edits is good and ground truth, but not particular combinations of exposure+wb because as we've seen they are interrelated.

That got me thinking how to express my ground truth in 'independent' way not allowing for inconsistencies.
And I think I have it: my params are not golden truth per se, but the inverted PICTURE that they produce IS.
In getting this picture I strive mostly for:
- middtones being sufficiently bright
- but not at the expense of clipping
- inverted picture having natural color balance in shadows/highlights

And what does characterize all of that params? HISTOGRAM!
That is the param-INVARIANT source of ground truth!

So why don't you tune the algo not for blindly matching my PARAMS, but for matching the HISTOGRAM that is obtained by APPLYING my params? I.e. your goal should be to bring respective histograms closer to my ground truth.

I've now have two folders of human annotations:
- 