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
		matmulKernel(out, x, w, n, d)
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

func startMatmulWorkers() {
	matmulStartOnce.Do(func() {
		for i := 0; i < matmulMaxWorkers; i++ {
			go func() {
				for job := range matmulJobs {
					matmulKernel(job.out, job.x, job.w, job.n, job.d)
					job.wg.Done()
				}
			}()
		}
	})
}
