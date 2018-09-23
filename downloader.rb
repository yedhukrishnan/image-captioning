# File downloader
# Downloads the list of files mentioned in the training TSV file

require 'open-uri'

file_name = 'Train%2FGCC-training.tsv'
data = open(file_name).read.split(/\n/)

info_file = File.open('data/info.txt', 'w+')

data[1..10000].each_with_index do |line, index|
  caption, url = line.split(/\t/)
  text = "#{index}, #{caption}, #{url}"
  puts text
  begin
    File.write "data/#{index}.jpg", open(url).read
    info_file.puts text
  rescue
    puts "ERROR: #{text}"
  end
end
