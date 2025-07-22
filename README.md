# 🚀 ArXivist

**The Ultimate arXiv Paper Download Tool**

ArXivist is a powerful, elegant CLI tool that downloads research papers from arXiv with smart duplicate detection and unlimited download capabilities. Perfect for researchers, ML engineers, and anyone who wants to build their own paper library.

## ✨ Features

- 🎯 **Smart Search** - Query by categories, keywords, or complex search terms
- 🔄 **Duplicate Prevention** - Never downloads the same paper twice
- 🎨 **Clean Filenames** - Sanitized filenames with paper IDs and titles
- ⚡ **Unlimited Downloads** - Download ALL papers by default (no artificial limits)
- 📁 **Custom Directories** - Organize papers wherever you want
- 🛡️ **Error Handling** - Robust error handling with detailed feedback
- 📊 **Progress Tracking** - See what's being downloaded and what's skipped

## 🚀 Quick Start

### Installation

```bash
git clone <your-repo-url>
cd arxivist
pip install -r requirements.txt
```

### Basic Usage

```bash
# Download ALL Machine Learning papers (default behavior)
python arxivist.py

# Download ALL Computer Vision papers
python arxivist.py --query "computer vision"

# Limit downloads to 50 papers
python arxivist.py --max 50

# Download to custom directory
python arxivist.py --dir ./my_papers
```

## 🎛️ Command Line Options

| Option | Short | Description | Default |
|--------|--------|-------------|---------|
| `--query` | `-q` | Search query or category | `cat:cs.LG OR cat:stat.ML` |
| `--max` | `-m` | Maximum papers to download | No limit (downloads ALL) |
| `--dir` | `-d` | Download directory | `./papers` |
| `--help` | `-h` | Show help message | - |
| `--version` | `-v` | Show version | - |

## 🔍 Search Examples

### By Category
```bash
# Machine Learning
python arxivist.py --query "cat:cs.LG"

# Computer Vision
python arxivist.py --query "cat:cs.CV"

# Natural Language Processing
python arxivist.py --query "cat:cs.CL"

# Artificial Intelligence
python arxivist.py --query "cat:cs.AI"
```

### By Keywords
```bash
# Neural Networks
python arxivist.py --query "neural networks"

# Deep Learning + Computer Vision
python arxivist.py --query "deep learning AND computer vision"

# Transformers
python arxivist.py --query "transformer OR attention mechanism"

# Recent papers only
python arxivist.py --query "cat:cs.LG AND submittedDate:[202401* TO *]"
```

### Advanced Queries
```bash
# Multiple categories
python arxivist.py --query "cat:cs.LG OR cat:cs.CV OR cat:cs.AI"

# Specific authors
python arxivist.py --query "au:Goodfellow OR au:LeCun"

# Title search
python arxivist.py --query "ti:GAN OR ti:Generative"
```

## 📁 File Organization

Downloaded papers are organized with the format:
```
papers/
├── 2301.07041_Attention_Is_All_You_Need.pdf
├── 2301.07042_Deep_Residual_Learning_for_Image_Recognition.pdf
└── 2301.07043_Generative_Adversarial_Networks.pdf
```

- **Format**: `{arxiv_id}_{sanitized_title}.pdf`
- **Smart Deduplication**: Automatically skips previously downloaded papers
- **Safe Filenames**: Invalid characters replaced with underscores

## 🎯 Use Cases

### 📚 Research Library
Build a comprehensive research library:
```bash
# Download all ML papers to your research folder
python arxivist.py --dir ~/Research/ML_Papers
```

### 🔬 Focused Research
Focus on specific research areas:
```bash
# Get all recent Transformer papers
python arxivist.py --query "transformer AND submittedDate:[2024* TO *]" --max 100
```

### 📊 Dataset Building
Create datasets for analysis:
```bash
# Download papers for meta-research
python arxivist.py --query "cat:cs.LG" --dir ./dataset/ml_papers
```

## 🛠️ Technical Details

### Dependencies
- `arxiv` - Official arXiv Python client
- `feedparser` - RSS/Atom feed parsing
- `requests` - HTTP library

### Paper Detection
ArXivist tracks downloaded papers by extracting arXiv IDs from filenames, ensuring:
- ✅ No duplicate downloads
- ✅ Resume capability after interruption
- ✅ Efficient batch processing

### Search Categories
Common arXiv categories for ML/AI research:
- `cs.LG` - Machine Learning
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.CL` - Computation and Language (NLP)
- `cs.AI` - Artificial Intelligence
- `cs.NE` - Neural and Evolutionary Computing
- `stat.ML` - Machine Learning (Statistics)

## ⚡ Performance Tips

1. **Incremental Downloads**: Run regularly to stay up-to-date with new papers
2. **Specific Queries**: Use targeted searches to avoid downloading irrelevant papers
3. **Directory Organization**: Use separate directories for different research areas
4. **Resume Capability**: Safely interrupt and resume downloads anytime

## 🚨 Rate Limiting

ArXivist respects arXiv's rate limits and includes built-in error handling. If you encounter rate limiting:
- Wait a few minutes and resume
- Consider using more specific queries to reduce load
- Download in smaller batches using `--max`

## 🤝 Contributing

Feel free to open issues or submit pull requests! Areas for improvement:
- Additional metadata extraction
- Better search query builder
- Paper categorization features
- Integration with reference managers

## 📄 License

MIT License - Feel free to use in your research and projects!

---

**Happy Paper Hunting!** 📖✨

*Built with ❤️ by WAZA.baby for the research community*
