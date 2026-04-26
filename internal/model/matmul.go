package model

import (
	"runtime"
	"sync"
)

const (
	matmulMinParallelOps = 1 << 20
	matmulRowsPerWorker  = 256
	matmulMaxWorkers     = 16
)

type matmulJob struct {
	out []float32
	x   []float32
	w   []float32
	n   int
	d   int
	wg  *sync.WaitGroup
}

var (
	matmulJobs          = make(chan matmulJob, matmulMaxWorkers)
	matmulStartOnce     sync.Once
	matmulWaitGroupPool = sync.Pool{New: func() any { return new(sync.WaitGroup) }}
)

func matmul(out []float32, x []float32, w []float32, n int, d int) {
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, d/matmulRowsPerWorker)
	if n*d < matmulMinParallelOps || workers < 2 {
		matmulF32(out, x, w, n, d)
		return
	}
	startMatmulWorkers()

	out = out[:d]
	x = x[:n]
	w = w[:d*n]

	rowsPerWorker := (d + workers - 1) / workers
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulJobs <- matmulJob{
			out: out[start:end],
			x:   x,
			w:   w[start*n : end*n],
			n:   n,
			d:   end - start,
			wg:  wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulBatch(out []float32, x []float32, w []float32, batch int, n int, d int) {
	if batch == 1 {
		matmul(out[:d], x[:n], w, n, d)
		return
	}
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, d/matmulRowsPerWorker)
	if batch*n*d < matmulMinParallelOps || workers < 2 {
		matmulBatchRows(out, x, w, batch, n, d, 0, d)
		return
	}

	out = out[:batch*d]
	x = x[:batch*n]
	w = w[:d*n]

	rowsPerWorker := (d + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		go func(row0 int, row1 int) {
			defer wg.Done()
			matmulBatchRows(out, x, w, batch, n, d, row0, row1)
		}(start, end)
	}
	wg.Wait()
}

func matmulBatchRows(out []float32, x []float32, w []float32, batch int, n int, d int, row0 int, row1 int) {
	for row := row0; row < row1; row++ {
		weight := w[row*n : (row+1)*n]
		for b := 0; b < batch; b++ {
			out[b*d+row] = dotF32(x[b*n:(b+1)*n], weight)
		}
	}
}

func startMatmulWorkers() {
	matmulStartOnce.Do(func() {
		for i := 0; i < matmulMaxWorkers; i++ {
			go func() {
				for job := range matmulJobs {
					matmulF32(job.out, job.x, job.w, job.n, job.d)
					job.wg.Done()
				}
			}()
		}
	})
}
