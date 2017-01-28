#define WARP_SIZE 32

/* Intra-warp local scan with K = 4 */
inline void intra_warp_local_scan(__global int *X, +) {
	//Compute lane ID within the warp
	int id = get_local_id(0) & 31; 

	//Load the elements
	e1 = X[id];
	e2 = X[id + 32];
	e3 = X[id + 64];
	e4 = X[id + 96];

	//Scan each row
	#pragma unroll
	for (int i = 1; i <= 32; i *= 2) {

	}
	#pragma unroll
	for (int i = 1; i <= 32; i *= 2) {

	}
	#pragma unroll
	for (int i = 1; i <= 32; i *= 2) {

	}
	#pragma unroll
	for (int i = 1; i <= 32; i *= 2) {

	}
}


//Almost the same as naive scan1Inclusive but doesn't need barriers
//and works only for size = WARP_SIZE
inline uint intra_warp_local_scan(uint idata, volatile __local uint *l_Data, uint size){
	uint lid = get_local_id(0);
    uint pos = 2 * lid - (lid & 31);
    l_Data[pos] = 0;
    pos += WARP_SIZE;
    l_Data[pos] = idata;

    if(WARP_SIZE >=  2) l_Data[pos] += l_Data[pos -  1];
    if(WARP_SIZE >=  4) l_Data[pos] += l_Data[pos -  2];
    if(WARP_SIZE >=  8) l_Data[pos] += l_Data[pos -  4];
    if(WARP_SIZE >= 16) l_Data[pos] += l_Data[pos -  8];
    if(WARP_SIZE >= 32) l_Data[pos] += l_Data[pos - 16];

    return l_Data[pos];
}