import cvbase as cvb
# to visualize a flow file
cvb.show_flow('img.flo')
# to visualize a loaded flow map
flow = np.random.rand(100, 100, 2).astype(np.float32)
cvb.show_flow(flow)
