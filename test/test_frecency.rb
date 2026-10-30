# frozen_string_literal: true

require_relative 'lib/common'

# Frecency behavior through the terminal and across invocations.
class TestFrecency < TestInteractive
  def test_manual_adjustments_refresh_ranking_and_survive_abort
    Dir.mktmpdir('fzf-frecency') do |dir|
      database = "--frecency-file=#{dir}/history.gob"
      options = ['--frecency', database, '--layout=reverse', '--query=file',
                 '--bind=ctrl-b:frecency-entry-buff,ctrl-n:first+frecency-entry-nerf,ctrl-r:first+frecency-entry-remove']
      writelines(%w[file-a file-b])
      tmux.send_keys "cat #{tempname} | #{fzf(*options)}", :Enter
      tmux.until { |lines| assert_equal ['> file-a', '  file-b'], lines.grep(/^[ >] file-/) }
      tmux.send_keys :Down, 'C-b'
      tmux.until { |lines| assert_equal(%w[file-b file-a], lines.grep(/^[ >] file-/).map { it[2..] }) }
      tmux.send_keys 'C-n'
      tmux.until { |lines| assert_equal(%w[file-a file-b], lines.grep(/^[ >] file-/).map { it[2..] }) }
      tmux.send_keys :Down, 'C-b'
      tmux.until { |lines| assert_equal(%w[file-b file-a], lines.grep(/^[ >] file-/).map { it[2..] }) }
      history = File.binread("#{dir}/history.gob")
      tmux.send_keys 'C-c'
      assert_empty fzf_output
      assert_equal history, File.binread("#{dir}/history.gob")

      # Abort must retain explicit adjustments without recording a selection.
      output = IO.popen("cat #{tempname} | #{fzf!('--frecency', database, '--filter=file')}", &:read)
      assert_equal %w[file-b file-a], output.lines(chomp: true)
      assert_equal history, File.binread("#{dir}/history.gob")

      tmux.send_keys "cat #{tempname} | #{fzf(*options)}", :Enter
      tmux.until { |lines| assert_includes lines, '> file-b' }
      tmux.send_keys 'C-r'
      tmux.until { |lines| assert_equal(%w[file-a file-b], lines.grep(/^[ >] file-/).map { it[2..] }) }
      tmux.send_keys 'C-c'
      assert_empty fzf_output
      output = IO.popen("cat #{tempname} | #{fzf!('--frecency', database, '--filter=file')}", &:read)
      assert_equal %w[file-a file-b], output.lines(chomp: true)
    end
  end

  def test_multi_accept_records_original_items
    Dir.mktmpdir('fzf-frecency') do |dir|
      database = "--frecency-file=#{dir}/history.gob"
      writelines(["\e[31mone file-a漢\e[0m", "\e[32mtwo file-b漢\e[0m", 'three file-c漢'])
      options = ['--frecency', database, '--ansi', '--multi', '--with-nth=2', '--accept-nth=1', '--query=file']
      tmux.send_keys "cat #{tempname} | #{fzf(*options)}", :Enter
      tmux.until { |lines| assert_equal 3, lines.match_count }
      tmux.send_keys :Up, :BTab, :BTab, :Enter
      assert_equal %w[two three], fzf_output_lines

      output = IO.popen("cat #{tempname} | #{fzf!('--frecency', database, '--ansi', '--filter=file')}", &:read)
      assert_equal ['two file-b漢', 'three file-c漢', 'one file-a漢'], output.lines(chomp: true)
      rows = IO.popen(fzf!(database, '--frecency-debug'), &:read).lines.grep(/file-/)
      assert_equal([1, 1], rows.map { it.split[1].to_i })
      assert_equal ['three file-c漢', 'two file-b漢'], rows.map { it.split.last(2).join(' ') }.sort
    end
  end
end
