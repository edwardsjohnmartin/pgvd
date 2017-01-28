__kernel scan_kernel(x, y, M, +, ...) {
	for (int i = get_group_id(0); i < M; i +=  get_local_size(0)) {
		intra_warp_local_scan(...);
		intra_block_local_scan(...);
		inter_block_comm(...);
		intra_block_global_scan(...);
	}
}