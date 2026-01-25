import os

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

outdir = os.path.join(desktop,'cameraOutput');
os.mkdir(outdir)

testdir = os.path.join(outdir,'cameraTesting')
os.mkdir(testdir)

savedir = os.path.join(outdir,'beeHotel');
os.mkdir(savedir)