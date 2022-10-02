void open_samples(std::string &dotfile, size_t zlevel,
		  std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p);
void close_samples(const std::string &file, std::string &dotfile, size_t overflows,
		   std::ofstream *outfile_p, boost::iostreams::filtering_streambuf<boost::iostreams::output> *outbuf_p);
