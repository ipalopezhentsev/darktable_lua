There are several types of dust when talking about film scanning using a DSLR camera:
- Dust on the film itself at the moment of scanning. This is what detect_dust.py has been doing so far. Appears in the random places for each film frame, because each film frame picks up its own random share of dust.
- Dust on the DSLR camera sensor. This one is completely unrelated to film, but it affects film scanning nevertheless, because film is scanned with apertures like f/11 and so any tiny dust collected on DSLR camera sensor becomes perfectly visible as blurry blobs on top of scans.

This spec is about removing DSLR sensor dust.  It's very different from dust on film frames:
- For one scanning session (one script invocation), sensor dust is in exactly the same places on all frames. It's because it's stuck on camera sensor and doesn't move from frame to frame. So this is one property for its detection. At the same time, after scanning each film frame is cropped differently and also can be further transformed with lens correction plugin for example. 
- It's mostly visible on low detail areas like bokeh or blue skies or water
- Example is given in file sensor_dust.png. There you see gray winter sky and one blurred sensor dust in the center