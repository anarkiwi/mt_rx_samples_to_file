#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include <iostream>
#include "sample_writer.h"


std::string get_prefix_file(const std::string &file, const std::string &prefix) {
    boost::filesystem::path orig_path(file);
    std::string basename(orig_path.filename().c_str());
    std::string dirname(boost::filesystem::canonical(orig_path.parent_path()).c_str());
    return dirname + "/" + prefix + basename;
}


std::string get_dotfile(const std::string &file) {
    return get_prefix_file(file, ".");
}


SampleWriter::SampleWriter() {
    outbuf_p = new boost::iostreams::filtering_streambuf<boost::iostreams::output>();
    out_p = new std::ostream(outbuf_p);
}


SampleWriter::~SampleWriter() {
    free(out_p);
    free(outbuf_p);
}


void SampleWriter::write(const char *data, size_t len) {
    if (!outbuf_p->empty()) {
	out_p->write(data, len);
    }
}


void SampleWriter::open(const std::string &file, size_t zlevel) {
    file_ = file;
    dotfile_ = get_dotfile(file_);
    orig_path_ = boost::filesystem::path(file_);
    std::cout << "opening " << dotfile_ << std::endl;
    outfile.open(dotfile_.c_str(), std::ofstream::binary);
    if (!outfile.is_open()) {
	throw std::runtime_error(dotfile_ + " could not be opened");
    }
    if (orig_path_.has_extension()) {
	if (orig_path_.extension() == ".gz") {
	    std::cout << "writing gzip compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::gzip_compressor(
			       boost::iostreams::gzip_params(zlevel)));
	} else if (orig_path_.extension() == ".zst") {
	    std::cout << "writing zstd compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::zstd_compressor(
			       boost::iostreams::zstd_params(zlevel)));
	} else {
	    std::cout << "writing uncompressed output" << std::endl;
	}
    }
    outbuf_p->push(outfile);
}


void SampleWriter::close(size_t overflows) {
    if (outfile.is_open()) {
	std::string dirname(boost::filesystem::canonical(orig_path_.parent_path()).c_str());

	std::cout << "closing " << file_ << std::endl;
	boost::iostreams::close(*outbuf_p);
	outfile.close();

	if (overflows) {
	    std::string overflow_name = dirname + "/overflow-" + file_;
	    rename(dotfile_.c_str(), overflow_name.c_str());
	} else {
	    rename(dotfile_.c_str(), file_.c_str());
	}
    }
}
