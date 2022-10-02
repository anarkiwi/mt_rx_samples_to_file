#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zstd.hpp>


std::string get_prefix_file(const std::string &file, const std::string &prefix) {
    boost::filesystem::path orig_path(file);
    std::string basename(orig_path.filename().c_str());
    std::string dirname(boost::filesystem::canonical(orig_path.parent_path()).c_str());
    return dirname + "/" + prefix + basename;
}


std::string get_dotfile(const std::string &file) {
    return get_prefix_file(file, ".");
}


void open_samples(const std::string &file, size_t zlevel,
		  std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p) {
    std::string dotfile = get_dotfile(file);
    std::cout << "opening " << dotfile << std::endl;
    boost::filesystem::path orig_path(dotfile);
    outfile_p->open(dotfile.c_str(), std::ofstream::binary);
    if (!outfile_p->is_open()) {
	throw std::runtime_error(dotfile + " could not be opened");
    }
    if (orig_path.has_extension()) {
	if (orig_path.extension() == ".gz") {
	    std::cout << "writing gzip compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::gzip_compressor(
		boost::iostreams::gzip_params(zlevel)));
	} else if (orig_path.extension() == ".zst") {
	    std::cout << "writing zstd compressed output" << std::endl;
	    outbuf_p->push(boost::iostreams::zstd_compressor(
		boost::iostreams::zstd_params(zlevel)));
	} else {
	    std::cout << "writing uncompressed output" << std::endl;
	}
    }
    outbuf_p->push(*outfile_p);
}


void close_samples(const std::string &file, size_t overflows,
		   std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p) {
    if (outfile_p->is_open()) {
	boost::filesystem::path orig_path(file);
	std::string dirname(boost::filesystem::canonical(orig_path.parent_path()).c_str());

	std::cout << "closing " << file << std::endl;
	boost::iostreams::close(*outbuf_p);
	outfile_p->close();

        std::string dotfile = get_dotfile(file);
	if (overflows) {
	    std::string overflow_name = dirname + "/overflow-" + file;
	    rename(dotfile.c_str(), overflow_name.c_str());
	} else {
	    rename(dotfile.c_str(), file.c_str());
	}
    }
}
