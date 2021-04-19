import matplotlib.pyplot as plt
from math import log

num_index = [761757, 382409, 154826, 78962, 41011, 22051, 12563]
arcface_public = [0.214198, 0.237164, 0.265617, 0.286468, 0.302344, 0.317417, 0.331209]
arcface_private = [0.233212, 0.259910, 0.281880, 0.304259, 0.325231, 0.344080, 0.357800]
ge2e_public = [0.061897, 0.077156, 0.099128, 0.120826, 0.137490, 0.156727, 0.173768]
ge2e_private = [0.071373, 0.086414, 0.110432, 0.130272, 0.149303, 0.168435, 0.181976]
unet_public = [0.017318, 0.023210, 0.030575, 0.038860, 0.049006, 0.054817, 0.063312]
unet_private = [0.019721, 0.024599, 0.030248, 0.036796, 0.043765, 0.054095, 0.064802]

arcface_public = [i/arcface_public[-1] for i in arcface_public]
arcface_private = [i/arcface_private[-1] for i in arcface_private]
ge2e_public = [i/ge2e_public[-1] for i in ge2e_public]
ge2e_private = [i/ge2e_private[-1] for i in ge2e_private]
unet_public = [i/unet_public[-1] for i in unet_public]
unet_private = [i/unet_private[-1] for i in unet_private]

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10,4))
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(221, sharey = ax1)

ax1.plot(num_index, arcface_public, 's-', color='#d62728', markersize=7, label="ArcFace")
ax1.plot(num_index, ge2e_public, 'o-', color='#1f77b4', markersize=7, label="GE2E")
ax1.plot(num_index, unet_public, '^-', color='#ff7f0e', markersize=7, label="UNet")
ax1.title.set_text("mAP (Public)")
# plt.savefig("mAP_public.png", dpi = 300, transparent = False)

# plt.figure(2)
ax2.plot(num_index, arcface_private, 'rs-', color='#d62728', markersize=7, label="ArcFace")
ax2.plot(num_index, ge2e_private, 'bo-', color='#1f77b4', markersize=7, label="GE2E")
ax2.plot(num_index, unet_private, 'm^-', color='#ff7f0e', markersize=7, label="UNet")
ax2.title.set_text("mAP (Private)")
plt.legend()
plt.savefig("mAP.pdf", dpi = 300, transparent = True)

# plt.figure(3)