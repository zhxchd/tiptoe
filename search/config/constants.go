package config

import (
  "encoding/json"
  "math"
  "os"
  "path/filepath"
)

type Config struct {
  preamble    string
  imageSearch bool
  overrides   *FileConfig
}

type FileConfig struct {
  EmbeddingDim                    int  `json:"embedding_dim"`
  SlotBits                        int  `json:"slot_bits"`
  TotalNumClusters                int  `json:"total_num_clusters"`
  MaxEmbeddingServers             int  `json:"max_embedding_servers"`
  MaxUrlServers                   *int `json:"max_url_servers"`
  EmbeddingsHintSize              int  `json:"embeddings_hint_size"`
  SimplePIREmbeddingsRecordLength int  `json:"simplepir_embeddings_record_length"`
}

func MakeConfig(preambleStr string, images bool) *Config {
  c := Config{
	preamble: preambleStr,
	imageSearch: images,
  }
  c.loadOverrides()
  return &c
}

func (c *Config) loadOverrides() {
  f, err := os.Open(filepath.Join(c.preamble, "tiptoe_config.json"))
  if err != nil {
    return
  }
  defer f.Close()

  overrides := new(FileConfig)
  if err := json.NewDecoder(f).Decode(overrides); err != nil {
    panic(err)
  }
  c.overrides = overrides
}

func (c *Config) PREAMBLE() string {
  return c.preamble
}

func (c *Config) IMAGE_SEARCH() bool {
  return c.imageSearch
}

// TODO: Fix to be more accurate
func (c *Config) DEFAULT_EMBEDDINGS_HINT_SZ() uint64 {
  if c.overrides != nil && c.overrides.EmbeddingsHintSize > 0 {
    return uint64(c.overrides.EmbeddingsHintSize)
  }
  if !c.imageSearch {
    return 500
  } else {
    return 900
  }
}

func DEFAULT_URL_HINT_SZ() uint64 {
  return 100
}

func (c *Config) EMBEDDINGS_DIM() uint64 {
  if c.overrides != nil && c.overrides.EmbeddingDim > 0 {
    return uint64(c.overrides.EmbeddingDim)
  }
  if !c.imageSearch {
    return 192
  } else {
    return 384
  }
}

func SLOT_BITS() uint64 {
  return 5
}

func (c *Config) SLOT_BITS() uint64 {
  if c.overrides != nil && c.overrides.SlotBits > 0 {
    return uint64(c.overrides.SlotBits)
  }
  return SLOT_BITS()
}

func (c *Config) TOTAL_NUM_CLUSTERS() int {
  if c.overrides != nil && c.overrides.TotalNumClusters > 0 {
    return c.overrides.TotalNumClusters
  }
  if !c.imageSearch {
    return 25196
  } else {
    return 42528
  }
}

// Round up (# clusters / # embedding servers)
func (c *Config) EMBEDDINGS_CLUSTERS_PER_SERVER() int {
  clustersPerServer := float64(c.TOTAL_NUM_CLUSTERS()) / float64(c.MAX_EMBEDDINGS_SERVERS())
  return int(math.Ceil(clustersPerServer))
}

func (c *Config) MAX_EMBEDDINGS_SERVERS() int {
  if c.overrides != nil && c.overrides.MaxEmbeddingServers > 0 {
    return c.overrides.MaxEmbeddingServers
  }
  if !c.imageSearch {
    return 80
  } else {
    return 160
  }
}

// Round up (# clusters / # url servers)
func (c *Config) URL_CLUSTERS_PER_SERVER() int {
  clustersPerServer := float64(c.TOTAL_NUM_CLUSTERS()) / float64(c.MAX_URL_SERVERS())
  return int(math.Ceil(clustersPerServer))
}

func (c *Config) MAX_URL_SERVERS() int {
  if c.overrides != nil && c.overrides.MaxUrlServers != nil {
    return *c.overrides.MaxUrlServers
  }
  if !c.imageSearch {
    return 8
  } else {
    return 16
  }
}

func (c *Config) SIMPLEPIR_EMBEDDINGS_RECORD_LENGTH() int {
  if c.overrides != nil && c.overrides.SimplePIREmbeddingsRecordLength > 0 {
    return c.overrides.SimplePIREmbeddingsRecordLength
  }
  if !c.imageSearch {
    return 17
  } else {
    return 15
  }
}
