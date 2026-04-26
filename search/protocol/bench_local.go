package protocol

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
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
	Benchmark       string            `json:"benchmark"`
	IncludesURLPIR  bool              `json:"includes_url_pir"`
	GOMAXPROCS      int               `json:"gomaxprocs"`
	ServerArtifact  string            `json:"server_artifact"`
	Notes           string            `json:"notes"`
	Corpus          CorpusBenchReport `json:"corpus"`
	Setup           SetupBenchReport  `json:"setup"`
	Preprocessing   PhaseBenchReport  `json:"preprocessing"`
	OnlineEmbedding PhaseBenchReport  `json:"online_embedding"`
	PerQueryTotals  TotalsBenchReport `json:"per_query_totals"`
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
}

type PhaseBenchReport struct {
	QueryMB                   float64 `json:"query_mb"`
	AnswerMB                  float64 `json:"answer_mb"`
	CommunicationMB           float64 `json:"communication_mb"`
	ClientQuerySecondsMean    float64 `json:"client_query_seconds_mean"`
	ClientQuerySecondsTotal   float64 `json:"client_query_seconds_total"`
	ClientCoreSecondsMean     float64 `json:"client_core_seconds_mean"`
	ClientCoreSecondsTotal    float64 `json:"client_core_seconds_total"`
	ServerAnswerSecondsMean   float64 `json:"server_answer_seconds_mean"`
	ServerAnswerSecondsTotal  float64 `json:"server_answer_seconds_total"`
	ServerCoreSecondsMean     float64 `json:"server_core_seconds_mean"`
	ServerCoreSecondsTotal    float64 `json:"server_core_seconds_total"`
	ClientRecoverSecondsMean  float64 `json:"client_recover_seconds_mean"`
	ClientRecoverSecondsTotal float64 `json:"client_recover_seconds_total"`
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

func BenchEmbeddingsLocal(conf *config.Config, numQueries int, outputPath string) {
	if numQueries <= 0 {
		panic("numQueries must be positive")
	}
	if conf.MAX_EMBEDDINGS_SERVERS() != 1 {
		panic("local embedding benchmark expects exactly one embedding server")
	}
	gob.Register(corpus.Params{})
	gob.Register(database.ClusterMap{})
	runtime.GOMAXPROCS(1)

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
	client.Setup(server.hint)
	clientSetupSeconds := time.Since(clientSetupStart).Seconds()

	var preprocessing PhaseBenchReport
	var online PhaseBenchReport

	for i := 0; i < numQueries; i++ {
		preprocessQueryStart := time.Now()
		hintQuery := client.PreprocessQuery()
		preprocessing.ClientQuerySecondsTotal += time.Since(preprocessQueryStart).Seconds()

		if i == 0 {
			preprocessing.QueryMB = utils.MessageSizeMB(*hintQuery)
		}

		preprocessServerStart := time.Now()
		hintAnswer := hintServer.HintAnswer(hintQuery)
		preprocessing.ServerAnswerSecondsTotal += time.Since(preprocessServerStart).Seconds()

		if i == 0 {
			preprocessing.AnswerMB = utils.MessageSizeMB(*hintAnswer)
		}

		preprocessRecoverStart := time.Now()
		client.ProcessHintApply(&UnderhoodAnswer{EmbAnswer: *hintAnswer})
		preprocessing.ClientRecoverSecondsTotal += time.Since(preprocessRecoverStart).Seconds()

		emb := embeddings.RandomEmbedding(client.params.EmbeddingSlots, (1 << (client.params.SlotBits - 1)))
		cluster := uint64(i % client.NumClusters())

		queryStart := time.Now()
		query := client.QueryEmbeddings(emb, cluster)
		online.ClientQuerySecondsTotal += time.Since(queryStart).Seconds()

		if i == 0 {
			online.QueryMB = utils.MessageSizeMB(*query)
		}

		serverStart := time.Now()
		answer := server.embeddingsServer.Answer(query)
		online.ServerAnswerSecondsTotal += time.Since(serverStart).Seconds()

		if i == 0 {
			online.AnswerMB = utils.MessageSizeMB(*answer)
		}

		recoverStart := time.Now()
		client.ReconstructEmbeddingsWithinCluster(answer, cluster)
		online.ClientRecoverSecondsTotal += time.Since(recoverStart).Seconds()
	}

	fillPhaseAverages := func(p *PhaseBenchReport) {
		p.CommunicationMB = p.QueryMB + p.AnswerMB
		p.ClientQuerySecondsMean = p.ClientQuerySecondsTotal / float64(numQueries)
		p.ServerAnswerSecondsMean = p.ServerAnswerSecondsTotal / float64(numQueries)
		p.ServerCoreSecondsMean = p.ServerAnswerSecondsMean
		p.ServerCoreSecondsTotal = p.ServerAnswerSecondsTotal
		p.ClientRecoverSecondsMean = p.ClientRecoverSecondsTotal / float64(numQueries)
		p.ClientCoreSecondsTotal = p.ClientQuerySecondsTotal + p.ClientRecoverSecondsTotal
		p.ClientCoreSecondsMean = p.ClientCoreSecondsTotal / float64(numQueries)
	}
	fillPhaseAverages(&preprocessing)
	fillPhaseAverages(&online)

	clientTotal := preprocessing.ClientQuerySecondsTotal +
		preprocessing.ClientRecoverSecondsTotal +
		online.ClientQuerySecondsTotal +
		online.ClientRecoverSecondsTotal
	serverTotal := preprocessing.ServerCoreSecondsTotal + online.ServerCoreSecondsTotal
	corpusParamsMB := utils.MessageSizeMB(server.hint.CParams)
	embeddingHintMB := utils.MessageSizeMB(server.hint.EmbeddingsHint)
	embeddingIndexMapMB := utils.MessageSizeMB(server.hint.EmbeddingsIndexMap)

	report := LocalEmbeddingBenchReport{
		NumQueries:     numQueries,
		Benchmark:      "tiptoe_embedding_local",
		IncludesURLPIR: false,
		GOMAXPROCS:     runtime.GOMAXPROCS(0),
		ServerArtifact: serverArtifact,
		Notes:          "ANNS/embedding phase only. URL/PIR is not run. Client and server core-seconds equal wall-clock seconds because this command forces GOMAXPROCS=1.",
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
			ClientSetupCoreSeconds:    clientSetupSeconds,
			HintTotalMB:               corpusParamsMB + embeddingHintMB + embeddingIndexMapMB,
			CorpusParamsMB:            corpusParamsMB,
			EmbeddingHintMB:           embeddingHintMB,
			EmbeddingIndexMapMB:       embeddingIndexMapMB,
		},
		Preprocessing:   preprocessing,
		OnlineEmbedding: online,
		PerQueryTotals: TotalsBenchReport{
			CommunicationMB:        preprocessing.CommunicationMB + online.CommunicationMB,
			ServerCoreSecondsMean:  serverTotal / float64(numQueries),
			ServerCoreSecondsTotal: serverTotal,
			ClientCoreSecondsMean:  clientTotal / float64(numQueries),
			ClientCoreSecondsTotal: clientTotal,
			ClientSecondsMean:      clientTotal / float64(numQueries),
			ClientSecondsTotal:     clientTotal,
		},
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
