package protocol

import (
	"bufio"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

import (
	"github.com/ahenzinger/underhood/underhood"
	"github.com/henrycg/simplepir/matrix"
)

import (
	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/corpus"
	"github.com/ahenzinger/tiptoe/search/database"
	"github.com/ahenzinger/tiptoe/search/embeddings"
	"github.com/ahenzinger/tiptoe/search/utils"
)

type LocalEmbeddingBenchReport struct {
	NumQueries      int               `json:"num_queries"`
	RetrievalDepth  int               `json:"retrieval_depth"`
	Benchmark       string            `json:"benchmark"`
	IncludesURLPIR  bool              `json:"includes_url_pir"`
	GOMAXPROCS      int               `json:"gomaxprocs"`
	ServerArtifact  string            `json:"server_artifact"`
	QueryFile       string            `json:"query_file"`
	Notes           string            `json:"notes"`
	Corpus          CorpusBenchReport `json:"corpus"`
	Setup           SetupBenchReport  `json:"setup"`
	Preprocessing   PhaseBenchReport  `json:"preprocessing"`
	OnlineEmbedding PhaseBenchReport  `json:"online_embedding"`
	PerQueryTotals  TotalsBenchReport `json:"per_query_totals"`
	QueryClusters   []int             `json:"query_clusters"`
	RetrievedIDs    [][]uint64        `json:"retrieved_ids"`
}

type CorpusBenchReport struct {
	NumDocs        uint64 `json:"num_docs"`
	NumClusters    int    `json:"num_clusters"`
	EmbeddingSlots uint64 `json:"embedding_slots"`
	SlotBits       uint64 `json:"slot_bits"`
}

type SetupBenchReport struct {
	ServerStateLoadSeconds    float64 `json:"server_state_load_seconds"`
	HintProcessorSetupSeconds float64 `json:"hint_processor_setup_seconds"`
	ClientSetupSeconds        float64 `json:"client_setup_seconds"`
	ClientSetupCoreSeconds    float64 `json:"client_setup_core_seconds"`
	HintTotalMB               float64 `json:"hint_total_mb"`
	CorpusParamsMB            float64 `json:"corpus_params_mb"`
	EmbeddingHintMB           float64 `json:"embedding_hint_mb"`
	EmbeddingIndexMapMB       float64 `json:"embedding_index_map_mb"`
	CentroidsMB               float64 `json:"centroids_mb"`
}

type PhaseBenchReport struct {
	QueryMB                       float64 `json:"query_mb"`
	AnswerMB                      float64 `json:"answer_mb"`
	CommunicationMB               float64 `json:"communication_mb"`
	ClientQuerySecondsMean        float64 `json:"client_query_seconds_mean"`
	ClientQuerySecondsTotal       float64 `json:"client_query_seconds_total"`
	ClientQueryCoreSecondsMean    float64 `json:"client_query_core_seconds_mean"`
	ClientQueryCoreSecondsTotal   float64 `json:"client_query_core_seconds_total"`
	ClientCoreSecondsMean         float64 `json:"client_core_seconds_mean"`
	ClientCoreSecondsTotal        float64 `json:"client_core_seconds_total"`
	ServerAnswerSecondsMean       float64 `json:"server_answer_seconds_mean"`
	ServerAnswerSecondsTotal      float64 `json:"server_answer_seconds_total"`
	ServerAnswerCoreSecondsMean   float64 `json:"server_answer_core_seconds_mean"`
	ServerAnswerCoreSecondsTotal  float64 `json:"server_answer_core_seconds_total"`
	ServerCoreSecondsMean         float64 `json:"server_core_seconds_mean"`
	ServerCoreSecondsTotal        float64 `json:"server_core_seconds_total"`
	ClientRecoverSecondsMean      float64 `json:"client_recover_seconds_mean"`
	ClientRecoverSecondsTotal     float64 `json:"client_recover_seconds_total"`
	ClientRecoverCoreSecondsMean  float64 `json:"client_recover_core_seconds_mean"`
	ClientRecoverCoreSecondsTotal float64 `json:"client_recover_core_seconds_total"`
}

type TotalsBenchReport struct {
	CommunicationMB        float64 `json:"communication_mb"`
	ServerCoreSecondsMean  float64 `json:"server_core_seconds_mean"`
	ServerCoreSecondsTotal float64 `json:"server_core_seconds_total"`
	ClientCoreSecondsMean  float64 `json:"client_core_seconds_mean"`
	ClientCoreSecondsTotal float64 `json:"client_core_seconds_total"`
	ClientSecondsMean      float64 `json:"client_seconds_mean"`
	ClientSecondsTotal     float64 `json:"client_seconds_total"`
}

type scoredDoc struct {
	ID    uint64
	Score int
}

func BenchEmbeddingsLocal(conf *config.Config, numQueries int, outputPath string, queryPath string, retrievalDepth int) {
	if numQueries <= 0 {
		panic("numQueries must be positive")
	}
	if retrievalDepth <= 0 {
		panic("retrievalDepth must be positive")
	}
	if conf.MAX_EMBEDDINGS_SERVERS() != 1 {
		panic("local embedding benchmark expects exactly one embedding server")
	}
	gob.Register(corpus.Params{})
	gob.Register(database.ClusterMap{})

	centroidPath := conf.PREAMBLE() + "/centroids.txt"
	centroids := readCentroids(centroidPath, int(conf.EMBEDDINGS_DIM()))
	queries := readFbin(queryPath, numQueries, int(conf.EMBEDDINGS_DIM()))
	numQueries = len(queries)

	serverArtifact := conf.EmbeddingServerLog(0)
	serverLoadStart := time.Now()
	server := NewServerFromFile(serverArtifact)
	serverLoadSeconds := time.Since(serverLoadStart).Seconds()

	hintSetupStart := time.Now()
	hintServer := underhood.NewServerHintOnly[matrix.Elem64](&server.hint.EmbeddingsHint.Hint)
	hintProcessorSetupSeconds := time.Since(hintSetupStart).Seconds()
	defer hintServer.Free()

	client := NewClient(false /* coordinator */)
	clientSetupStart := time.Now()
	clientSetupCPUStart := cpuNow()
	client.Setup(server.hint)
	clientSetupSeconds := time.Since(clientSetupStart).Seconds()
	clientSetupCoreSeconds := (cpuNow() - clientSetupCPUStart).Seconds()

	var preprocessing PhaseBenchReport
	var online PhaseBenchReport
	queryClusters := make([]int, 0, numQueries)
	retrievedIDs := make([][]uint64, 0, numQueries)
	clusterDocIDs := make(map[int][]uint64)

	for i := 0; i < numQueries; i++ {
		preprocessQueryStart := time.Now()
		preprocessQueryCPUStart := cpuNow()
		hintQuery := client.PreprocessQuery()
		preprocessing.ClientQuerySecondsTotal += time.Since(preprocessQueryStart).Seconds()
		preprocessing.ClientQueryCoreSecondsTotal += (cpuNow() - preprocessQueryCPUStart).Seconds()

		if i == 0 {
			preprocessing.QueryMB = utils.MessageSizeMB(*hintQuery)
		}

		preprocessServerStart := time.Now()
		preprocessServerCPUStart := cpuNow()
		hintAnswer := hintServer.HintAnswer(hintQuery)
		preprocessing.ServerAnswerSecondsTotal += time.Since(preprocessServerStart).Seconds()
		preprocessing.ServerAnswerCoreSecondsTotal += (cpuNow() - preprocessServerCPUStart).Seconds()

		if i == 0 {
			preprocessing.AnswerMB = utils.MessageSizeMB(*hintAnswer)
		}

		preprocessRecoverStart := time.Now()
		preprocessRecoverCPUStart := cpuNow()
		client.ProcessHintApply(&UnderhoodAnswer{EmbAnswer: *hintAnswer})
		preprocessing.ClientRecoverSecondsTotal += time.Since(preprocessRecoverStart).Seconds()
		preprocessing.ClientRecoverCoreSecondsTotal += (cpuNow() - preprocessRecoverCPUStart).Seconds()

		queryStart := time.Now()
		queryCPUStart := cpuNow()
		emb := quantizeQuery(queries[i], float32(32.0), conf.SLOT_BITS())
		cluster := uint64(nearestCentroid(centroids, emb))
		query := client.QueryEmbeddings(emb, cluster)
		online.ClientQuerySecondsTotal += time.Since(queryStart).Seconds()
		online.ClientQueryCoreSecondsTotal += (cpuNow() - queryCPUStart).Seconds()
		queryClusters = append(queryClusters, int(cluster))

		if i == 0 {
			online.QueryMB = utils.MessageSizeMB(*query)
		}

		serverStart := time.Now()
		serverCPUStart := cpuNow()
		answer := server.embeddingsServer.Answer(query)
		online.ServerAnswerSecondsTotal += time.Since(serverStart).Seconds()
		online.ServerAnswerCoreSecondsTotal += (cpuNow() - serverCPUStart).Seconds()

		if i == 0 {
			online.AnswerMB = utils.MessageSizeMB(*answer)
		}

		recoverStart := time.Now()
		recoverCPUStart := cpuNow()
		scores := client.ReconstructEmbeddingsWithinCluster(answer, cluster)
		docIDs, ok := clusterDocIDs[int(cluster)]
		if !ok {
			docIDs = readClusterDocIDs(conf.TxtCorpus(int(cluster)))
			clusterDocIDs[int(cluster)] = docIDs
		}
		retrievedIDs = append(retrievedIDs, topDocIDs(docIDs, scores, server.hint.EmbeddingsHint.Info.P(), retrievalDepth))
		online.ClientRecoverSecondsTotal += time.Since(recoverStart).Seconds()
		online.ClientRecoverCoreSecondsTotal += (cpuNow() - recoverCPUStart).Seconds()
	}

	fillPhaseAverages := func(p *PhaseBenchReport) {
		p.CommunicationMB = p.QueryMB + p.AnswerMB
		p.ClientQuerySecondsMean = p.ClientQuerySecondsTotal / float64(numQueries)
		p.ClientQueryCoreSecondsMean = p.ClientQueryCoreSecondsTotal / float64(numQueries)
		p.ServerAnswerSecondsMean = p.ServerAnswerSecondsTotal / float64(numQueries)
		p.ServerAnswerCoreSecondsMean = p.ServerAnswerCoreSecondsTotal / float64(numQueries)
		p.ServerCoreSecondsMean = p.ServerAnswerCoreSecondsMean
		p.ServerCoreSecondsTotal = p.ServerAnswerCoreSecondsTotal
		p.ClientRecoverSecondsMean = p.ClientRecoverSecondsTotal / float64(numQueries)
		p.ClientRecoverCoreSecondsMean = p.ClientRecoverCoreSecondsTotal / float64(numQueries)
		p.ClientCoreSecondsTotal = p.ClientQueryCoreSecondsTotal + p.ClientRecoverCoreSecondsTotal
		p.ClientCoreSecondsMean = p.ClientCoreSecondsTotal / float64(numQueries)
	}
	fillPhaseAverages(&preprocessing)
	fillPhaseAverages(&online)

	clientCoreTotal := preprocessing.ClientCoreSecondsTotal + online.ClientCoreSecondsTotal
	clientWallTotal := preprocessing.ClientQuerySecondsTotal +
		preprocessing.ClientRecoverSecondsTotal +
		online.ClientQuerySecondsTotal +
		online.ClientRecoverSecondsTotal
	serverTotal := preprocessing.ServerCoreSecondsTotal + online.ServerCoreSecondsTotal
	corpusParamsMB := utils.MessageSizeMB(server.hint.CParams)
	embeddingHintMB := utils.MessageSizeMB(server.hint.EmbeddingsHint)
	embeddingIndexMapMB := utils.MessageSizeMB(server.hint.EmbeddingsIndexMap)
	centroidsMB := fileSizeMB(centroidPath)

	report := LocalEmbeddingBenchReport{
		NumQueries:     numQueries,
		RetrievalDepth: retrievalDepth,
		Benchmark:      "tiptoe_embedding_local",
		IncludesURLPIR: false,
		GOMAXPROCS:     runtime.GOMAXPROCS(0),
		ServerArtifact: serverArtifact,
		QueryFile:      queryPath,
		Notes:          "ANNS/embedding phase only. URL/PIR is not run. The benchmark uses real DEEP query embeddings, completes Tiptoe embedding retrieval, and records top document IDs. Wall-clock seconds and process CPU core-seconds are recorded separately.",
		Corpus: CorpusBenchReport{
			NumDocs:        client.params.NumDocs,
			NumClusters:    client.NumClusters(),
			EmbeddingSlots: client.params.EmbeddingSlots,
			SlotBits:       client.params.SlotBits,
		},
		Setup: SetupBenchReport{
			ServerStateLoadSeconds:    serverLoadSeconds,
			HintProcessorSetupSeconds: hintProcessorSetupSeconds,
			ClientSetupSeconds:        clientSetupSeconds,
			ClientSetupCoreSeconds:    clientSetupCoreSeconds,
			HintTotalMB:               corpusParamsMB + embeddingHintMB + embeddingIndexMapMB,
			CorpusParamsMB:            corpusParamsMB,
			EmbeddingHintMB:           embeddingHintMB,
			EmbeddingIndexMapMB:       embeddingIndexMapMB,
			CentroidsMB:               centroidsMB,
		},
		Preprocessing:   preprocessing,
		OnlineEmbedding: online,
		PerQueryTotals: TotalsBenchReport{
			CommunicationMB:        preprocessing.CommunicationMB + online.CommunicationMB,
			ServerCoreSecondsMean:  serverTotal / float64(numQueries),
			ServerCoreSecondsTotal: serverTotal,
			ClientCoreSecondsMean:  clientCoreTotal / float64(numQueries),
			ClientCoreSecondsTotal: clientCoreTotal,
			ClientSecondsMean:      clientWallTotal / float64(numQueries),
			ClientSecondsTotal:     clientWallTotal,
		},
		QueryClusters: queryClusters,
		RetrievedIDs:  retrievedIDs,
	}

	out, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		panic(err)
	}
	out = append(out, '\n')

	if outputPath == "" || outputPath == "-" {
		fmt.Print(string(out))
		return
	}
	if err := os.WriteFile(outputPath, out, 0644); err != nil {
		panic(err)
	}
}

func readFbin(path string, limit int, wantDim int) [][]float32 {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var nvecs int32
	var dim int32
	if err := binary.Read(f, binary.LittleEndian, &nvecs); err != nil {
		panic(err)
	}
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
		panic(err)
	}
	if int(dim) != wantDim {
		panic(fmt.Sprintf("query dimension mismatch: got %d, want %d", dim, wantDim))
	}
	n := int(nvecs)
	if limit < n {
		n = limit
	}
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float32, dim)
		if err := binary.Read(f, binary.LittleEndian, out[i]); err != nil {
			panic(err)
		}
	}
	return out
}

func quantizeQuery(vals []float32, scale float32, slotBits uint64) []int8 {
	bound := int(1 << (slotBits - 1))
	out := make([]int8, len(vals))
	for i, v := range vals {
		q := int(math.Round(float64(v * scale)))
		if q < -bound {
			q = -bound
		}
		if q > bound-1 {
			q = bound - 1
		}
		out[i] = int8(q)
	}
	return out
}

func readCentroids(path string, dim int) [][]float32 {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var centroids [][]float32
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != dim {
			panic(fmt.Sprintf("centroid dimension mismatch in %s: got %d, want %d", path, len(fields), dim))
		}
		row := make([]float32, dim)
		for i, field := range fields {
			val, err := strconv.ParseFloat(field, 32)
			if err != nil {
				panic(err)
			}
			row[i] = float32(val)
		}
		centroids = append(centroids, row)
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return centroids
}

func nearestCentroid(centroids [][]float32, emb []int8) int {
	best := 0
	bestScore := float32(math.Inf(-1))
	for i, centroid := range centroids {
		score := float32(0)
		for j, v := range emb {
			score += centroid[j] * float32(v)
		}
		if score > bestScore {
			bestScore = score
			best = i
		}
	}
	return best
}

func readClusterDocIDs(path string) []uint64 {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var ids []uint64
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line == corpus.SUBCLUSTER_DELIM {
			continue
		}
		before, _, ok := strings.Cut(line, " | ")
		if !ok {
			panic(fmt.Sprintf("bad cluster line in %s: %s", path, line))
		}
		id, err := strconv.ParseUint(before, 10, 64)
		if err != nil {
			panic(err)
		}
		ids = append(ids, id)
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return ids
}

func topDocIDs(docIDs []uint64, scores []uint64, mod uint64, depth int) []uint64 {
	n := len(scores)
	if len(docIDs) < n {
		n = len(docIDs)
	}
	docs := make([]scoredDoc, n)
	for i := 0; i < n; i++ {
		docs[i] = scoredDoc{
			ID:    docIDs[i],
			Score: embeddings.SmoothResult(scores[i], mod),
		}
	}
	sort.Slice(docs, func(i, j int) bool {
		if docs[i].Score == docs[j].Score {
			return docs[i].ID < docs[j].ID
		}
		return docs[i].Score > docs[j].Score
	})
	if depth > len(docs) {
		depth = len(docs)
	}
	out := make([]uint64, depth)
	for i := 0; i < depth; i++ {
		out[i] = docs[i].ID
	}
	return out
}

func fileSizeMB(path string) float64 {
	info, err := os.Stat(path)
	if err != nil {
		panic(err)
	}
	return float64(info.Size()) / (1024.0 * 1024.0)
}
